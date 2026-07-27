#include "prior_icp.hpp"
#include "scan_context.hpp"

#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>

#include <Eigen/Geometry>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace fs = std::filesystem;
namespace sc = fast_lio::scan_context;
namespace prior_icp = fast_lio::prior_icp;
using Clock = std::chrono::steady_clock;
using PointType = prior_icp::PointType;
using PointCloud = prior_icp::PointCloudXYZI;

namespace
{

struct Options
{
    std::string method;
    fs::path hypotheses_csv;
    fs::path query_dir;
    fs::path map_pcd;
    fs::path scd_path;
    fs::path output_csv;
    std::vector<int> budgets{1, 5, 10, 100};
};

struct Query
{
    int id = -1;
    int window = -1;
    bool truth_valid = false;
    double truth_x = 0.0;
    double truth_y = 0.0;
    fs::path cloud_path;
    Eigen::Vector3d up = Eigen::Vector3d::UnitZ();
};

struct Hypothesis
{
    int candidate_rank = -1;
    int candidate_index = -1;
    double distance = std::numeric_limits<double>::infinity();
    double yaw_shift_rad = 0.0;
    double root_shift_y = 0.0;
    double vertical_shift = 0.0;
};

struct QueryHypotheses
{
    double retrieval_ms = 0.0;
    std::vector<Hypothesis> values;
};

struct BudgetResult
{
    int candidate_count = 0;
    int coarse_converged = 0;
    int coarse_valid = 0;
    int fine_converged = 0;
    int fine_valid = 0;
    double coarse_ms = 0.0;
    double fine_ms = 0.0;
    bool accepted = false;
    double fitness = std::numeric_limits<double>::infinity();
    double overlap = 0.0;
    Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
    int best_seed_rank = -1;
    int best_candidate_index = -1;
};

std::vector<std::string> split(const std::string &line, char delimiter = ',')
{
    std::vector<std::string> fields;
    std::stringstream stream(line);
    std::string field;
    while (std::getline(stream, field, delimiter))
    {
        if (!field.empty() && field.back() == '\r')
            field.pop_back();
        fields.push_back(field);
    }
    return fields;
}

std::size_t column(const std::vector<std::string> &header, const std::string &name)
{
    const auto found = std::find(header.begin(), header.end(), name);
    if (found == header.end())
        throw std::runtime_error("missing CSV column: " + name);
    return static_cast<std::size_t>(found - header.begin());
}

std::vector<int> parseBudgets(const std::string &text)
{
    std::vector<int> budgets;
    for (const auto &field : split(text))
    {
        const int value = std::stoi(field);
        if (value <= 0)
            throw std::runtime_error("candidate budgets must be positive");
        budgets.push_back(value);
    }
    std::sort(budgets.begin(), budgets.end());
    budgets.erase(std::unique(budgets.begin(), budgets.end()), budgets.end());
    if (budgets.empty())
        throw std::runtime_error("no candidate budgets provided");
    return budgets;
}

Options parseOptions(int argc, char **argv)
{
    Options options;
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        auto value = [&]() -> std::string {
            if (++i >= argc)
                throw std::runtime_error("missing value after " + arg);
            return argv[i];
        };
        if (arg == "--method")
            options.method = value();
        else if (arg == "--hypotheses")
            options.hypotheses_csv = value();
        else if (arg == "--query-dir")
            options.query_dir = value();
        else if (arg == "--map")
            options.map_pcd = value();
        else if (arg == "--scd")
            options.scd_path = value();
        else if (arg == "--output")
            options.output_csv = value();
        else if (arg == "--budgets")
            options.budgets = parseBudgets(value());
        else
            throw std::runtime_error("unknown argument: " + arg);
    }
    if (options.method != "sc" && options.method != "scpp" && options.method != "updown")
        throw std::runtime_error("--method must be sc, scpp, or updown");
    if (options.query_dir.empty() || options.map_pcd.empty() ||
        options.scd_path.empty() || options.output_csv.empty())
        throw std::runtime_error("--query-dir, --map, --scd, and --output are required");
    if (options.method != "updown" && options.hypotheses_csv.empty())
        throw std::runtime_error("SC and SC++ require --hypotheses");
    return options;
}

std::unordered_map<int, Eigen::Vector3d> loadGravity(const fs::path &path)
{
    std::ifstream input(path);
    if (!input)
        throw std::runtime_error("cannot open gravity CSV: " + path.string());
    std::string line;
    std::getline(input, line);
    const auto header = split(line);
    const auto id_col = column(header, "query_id");
    const auto x_col = column(header, "up_x");
    const auto y_col = column(header, "up_y");
    const auto z_col = column(header, "up_z");
    std::unordered_map<int, Eigen::Vector3d> gravity;
    while (std::getline(input, line))
    {
        const auto fields = split(line);
        Eigen::Vector3d up(
            std::stod(fields[x_col]), std::stod(fields[y_col]), std::stod(fields[z_col]));
        if (up.allFinite() && up.squaredNorm() > 1e-12)
            gravity.emplace(std::stoi(fields[id_col]), up.normalized());
    }
    return gravity;
}

std::vector<Query> loadQueries(const fs::path &query_dir)
{
    const auto gravity = loadGravity(query_dir / "gravity.csv");
    std::ifstream input(query_dir / "metadata.csv");
    if (!input)
        throw std::runtime_error("cannot open query metadata");
    std::string line;
    std::getline(input, line);
    const auto header = split(line);
    const auto id_col = column(header, "query_id");
    const auto window_col = column(header, "window");
    const auto valid_col = column(header, "truth_valid");
    const auto x_col = column(header, "truth_x");
    const auto y_col = column(header, "truth_y");
    const auto file_col = column(header, "file");
    std::vector<Query> queries;
    while (std::getline(input, line))
    {
        const auto fields = split(line);
        Query query;
        query.id = std::stoi(fields[id_col]);
        query.window = std::stoi(fields[window_col]);
        query.truth_valid = fields[valid_col] == "True" || fields[valid_col] == "true";
        if (query.truth_valid)
        {
            query.truth_x = std::stod(fields[x_col]);
            query.truth_y = std::stod(fields[y_col]);
        }
        query.cloud_path = query_dir / fields[file_col];
        const auto found = gravity.find(query.id);
        if (found != gravity.end())
            query.up = found->second;
        queries.push_back(query);
    }
    return queries;
}

std::unordered_map<int, QueryHypotheses> loadHypotheses(const fs::path &path)
{
    std::ifstream input(path);
    if (!input)
        throw std::runtime_error("cannot open hypothesis CSV: " + path.string());
    std::string line;
    std::getline(input, line);
    const auto header = split(line);
    const auto query_col = column(header, "query_id");
    const auto rank_col = column(header, "candidate_rank");
    const auto index_col = column(header, "candidate_index");
    const auto distance_col = column(header, "distance");
    const auto yaw_col = column(header, "yaw_shift_rad");
    const auto root_col = column(header, "root_shift_y");
    const auto vertical_col = column(header, "vertical_shift");
    const auto time_col = column(header, "retrieval_ms");
    std::unordered_map<int, QueryHypotheses> grouped;
    while (std::getline(input, line))
    {
        const auto fields = split(line);
        const int query_id = std::stoi(fields[query_col]);
        Hypothesis hypothesis;
        hypothesis.candidate_rank = std::stoi(fields[rank_col]);
        hypothesis.candidate_index = std::stoi(fields[index_col]);
        hypothesis.distance = std::stod(fields[distance_col]);
        hypothesis.yaw_shift_rad = std::stod(fields[yaw_col]);
        hypothesis.root_shift_y = std::stod(fields[root_col]);
        hypothesis.vertical_shift = std::stod(fields[vertical_col]);
        auto &query = grouped[query_id];
        query.retrieval_ms = std::stod(fields[time_col]);
        query.values.push_back(hypothesis);
    }
    for (auto &[query_id, hypotheses] : grouped)
    {
        (void)query_id;
        std::sort(hypotheses.values.begin(), hypotheses.values.end(),
                  [](const Hypothesis &lhs, const Hypothesis &rhs) {
                      return lhs.candidate_rank < rhs.candidate_rank;
                  });
    }
    return grouped;
}

PointCloud::Ptr loadBin(const fs::path &path)
{
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input)
        throw std::runtime_error("cannot open query cloud: " + path.string());
    const auto bytes = input.tellg();
    if (bytes <= 0 || static_cast<std::size_t>(bytes) % (4 * sizeof(float)) != 0)
        throw std::runtime_error("invalid XYZI query cloud: " + path.string());
    input.seekg(0);
    std::vector<float> values(static_cast<std::size_t>(bytes) / sizeof(float));
    input.read(reinterpret_cast<char *>(values.data()), bytes);
    PointCloud::Ptr cloud(new PointCloud());
    cloud->reserve(values.size() / 4);
    for (std::size_t i = 0; i < values.size(); i += 4)
    {
        if (!std::isfinite(values[i]) || !std::isfinite(values[i + 1]) ||
            !std::isfinite(values[i + 2]))
            continue;
        PointType point;
        point.x = values[i];
        point.y = values[i + 1];
        point.z = values[i + 2];
        point.intensity = values[i + 3];
        point.normal_x = point.normal_y = point.normal_z = 0.0F;
        point.curvature = 0.0F;
        cloud->push_back(point);
    }
    cloud->width = static_cast<std::uint32_t>(cloud->size());
    cloud->height = 1;
    cloud->is_dense = true;
    return cloud;
}

PointCloud::Ptr voxelize(const PointCloud::Ptr &cloud, float leaf)
{
    PointCloud::Ptr output(new PointCloud());
    pcl::VoxelGrid<PointType> voxel;
    voxel.setLeafSize(leaf, leaf, leaf);
    voxel.setInputCloud(cloud);
    voxel.filter(*output);
    return output;
}

Eigen::Matrix3d poseRotation(const sc::Pose &pose)
{
    return (Eigen::AngleAxisd(pose.yaw, Eigen::Vector3d::UnitZ()) *
            Eigen::AngleAxisd(pose.pitch, Eigen::Vector3d::UnitY()) *
            Eigen::AngleAxisd(pose.roll, Eigen::Vector3d::UnitX()))
        .toRotationMatrix();
}

std::vector<Eigen::Matrix4f> makeSeeds(
    const std::string &method,
    const std::vector<Hypothesis> &hypotheses,
    int budget,
    const sc::Database &database,
    const Eigen::Matrix3d &gravity_rotation,
    std::vector<int> &candidate_indices)
{
    const int count = std::min(budget, static_cast<int>(hypotheses.size()));
    std::vector<Eigen::Matrix4f> seeds;
    seeds.reserve(count);
    candidate_indices.clear();
    candidate_indices.reserve(count);
    for (int i = 0; i < count; ++i)
    {
        const auto &hypothesis = hypotheses[i];
        if (hypothesis.candidate_index < 0 ||
            hypothesis.candidate_index >= static_cast<int>(database.entries().size()))
            continue;
        const auto &pose = database.entries()[hypothesis.candidate_index].pose;
        Eigen::Vector3d translation(pose.x, pose.y, pose.z + hypothesis.vertical_shift);
        Eigen::Matrix3d rotation;
        if (method == "updown")
        {
            const double seed_yaw = sc::makeCandidateSeedYaw(
                pose.canonical_yaw, hypothesis.yaw_shift_rad);
            rotation = Eigen::AngleAxisd(seed_yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix() *
                       gravity_rotation;
        }
        else
        {
            const Eigen::Matrix3d map_frame_rotation = poseRotation(pose);
            rotation = map_frame_rotation *
                       Eigen::AngleAxisd(-hypothesis.yaw_shift_rad,
                                         Eigen::Vector3d::UnitZ()).toRotationMatrix();
            if (method == "scpp" && std::abs(hypothesis.root_shift_y) > 1e-12)
            {
                translation += map_frame_rotation *
                               Eigen::Vector3d(0.0, hypothesis.root_shift_y, 0.0);
            }
        }
        seeds.push_back(prior_icp::makeSeedTransform(
            translation.x(), translation.y(), translation.z(), rotation));
        candidate_indices.push_back(hypothesis.candidate_index);
    }
    return seeds;
}

QueryHypotheses makeUpDownHypotheses(
    const PointCloud::Ptr &source,
    const Eigen::Matrix3d &gravity_rotation,
    const sc::Database &database)
{
    QueryHypotheses output;
    const auto started = Clock::now();
    PointCloud filtered;
    filtered.reserve(source->size());
    constexpr double blind_sq = 0.3 * 0.3;
    for (const auto &point : source->points)
    {
        const double xy_sq = static_cast<double>(point.x) * point.x +
                             static_cast<double>(point.y) * point.y;
        const bool in_blind_cylinder =
            xy_sq <= blind_sq && point.z >= -0.5F && point.z <= 2.0F;
        if (!in_blind_cylinder)
            filtered.push_back(point);
    }
    filtered.width = static_cast<std::uint32_t>(filtered.size());
    filtered.height = 1;
    filtered.is_dense = true;
    PointCloud canonical = sc::gravityCanonicalize(filtered, gravity_rotation);
    PointCloud::Ptr canonical_ptr(new PointCloud(std::move(canonical)));
    const PointCloud::Ptr query_cloud = voxelize(canonical_ptr, 0.25F);
    const auto candidates = database.query(*query_cloud, false);
    output.retrieval_ms = std::chrono::duration<double, std::milli>(
        Clock::now() - started).count();
    output.values.reserve(candidates.size());
    for (std::size_t rank = 0; rank < candidates.size(); ++rank)
    {
        const auto &candidate = candidates[rank];
        if (candidate.yaw_matches.empty())
            continue;
        const auto &yaw = candidate.yaw_matches.front();
        Hypothesis hypothesis;
        hypothesis.candidate_rank = static_cast<int>(rank) + 1;
        hypothesis.candidate_index = candidate.index;
        hypothesis.distance = yaw.distance;
        hypothesis.yaw_shift_rad = yaw.yaw_shift_rad;
        hypothesis.vertical_shift = yaw.vertical_shift;
        output.values.push_back(hypothesis);
    }
    return output;
}

BudgetResult evaluateBudget(
    const prior_icp::Config &config,
    const PointCloud::Ptr &source_coarse,
    const PointCloud::Ptr &source_fine,
    const PointCloud::Ptr &map_coarse,
    const PointCloud::Ptr &map_fine,
    const std::vector<Eigen::Matrix4f> &seeds,
    const std::vector<int> &candidate_indices,
    int refine_top_k)
{
    BudgetResult output;
    output.candidate_count = static_cast<int>(seeds.size());
    if (seeds.empty())
        return output;
    std::vector<int> all_seed_indices(seeds.size());
    for (int i = 0; i < static_cast<int>(seeds.size()); ++i)
        all_seed_indices[i] = i;

    auto started = Clock::now();
    const auto coarse = prior_icp::runStage(
        config, source_coarse, map_coarse, seeds, all_seed_indices,
        output.coarse_converged, output.coarse_valid);
    output.coarse_ms = std::chrono::duration<double, std::milli>(
        Clock::now() - started).count();
    if (coarse.empty())
        return output;

    std::vector<int> fine_seed_indices;
    const int fine_count = std::min(refine_top_k, static_cast<int>(coarse.size()));
    fine_seed_indices.reserve(fine_count);
    for (int i = 0; i < fine_count; ++i)
        fine_seed_indices.push_back(coarse[i].seed_index);

    started = Clock::now();
    const auto fine = prior_icp::runStage(
        config, source_fine, map_fine, seeds, fine_seed_indices,
        output.fine_converged, output.fine_valid);
    output.fine_ms = std::chrono::duration<double, std::milli>(
        Clock::now() - started).count();
    if (fine.empty())
        return output;

    const auto &best = fine.front();
    output.accepted = best.fitness <= 0.1;
    output.fitness = best.fitness;
    output.overlap = best.overlap;
    output.transform = best.transform;
    output.best_seed_rank = best.seed_index + 1;
    if (best.seed_index >= 0 && best.seed_index < static_cast<int>(candidate_indices.size()))
        output.best_candidate_index = candidate_indices[best.seed_index];
    return output;
}

}  // namespace

int main(int argc, char **argv)
{
    try
    {
        const Options options = parseOptions(argc, argv);
        const auto queries = loadQueries(options.query_dir);

        sc::Config database_config;
        database_config.candidate_top_k = *std::max_element(
            options.budgets.begin(), options.budgets.end());
        database_config.yaw_top_k = 1;
        database_config.distance_thresh = 2.0;
        sc::Database database(database_config);
        std::string database_error;
        if (!database.load(options.scd_path.string(), &database_error))
            throw std::runtime_error("cannot load SCD: " + database_error);

        std::unordered_map<int, QueryHypotheses> external_hypotheses;
        if (!options.hypotheses_csv.empty())
            external_hypotheses = loadHypotheses(options.hypotheses_csv);

        PointCloud::Ptr map_raw(new PointCloud());
        if (pcl::io::loadPCDFile<PointType>(options.map_pcd.string(), *map_raw) < 0 ||
            map_raw->empty())
            throw std::runtime_error("cannot load map PCD");
        const PointCloud::Ptr map_coarse = voxelize(map_raw, 0.5F);
        const PointCloud::Ptr map_fine = voxelize(map_raw, 0.3F);

        if (!options.output_csv.parent_path().empty())
            fs::create_directories(options.output_csv.parent_path());
        std::ofstream output(options.output_csv);
        if (!output)
            throw std::runtime_error("cannot open output CSV");
        output << "algorithm,query_id,window,budget,available_candidates,retrieval_ms,"
                  "downsample_ms,coarse_icp_ms,fine_icp_ms,icp_ms,total_ms,"
                  "coarse_converged,coarse_valid,fine_converged,fine_valid,accepted,"
                  "truth_success_2m,xy_error_m,fitness,overlap,best_seed_rank,"
                  "best_candidate_index\n";
        output << std::fixed << std::setprecision(9);

        prior_icp::Config icp_config;
        icp_config.max_iterations = 30;
        icp_config.max_corr_dist = 0.5;
        icp_config.min_overlap_ratio = 0.5;

        int evaluated = 0;
        for (const auto &query : queries)
        {
            if (!query.truth_valid)
                continue;
            const PointCloud::Ptr source = loadBin(query.cloud_path);
            Eigen::Matrix3d gravity_rotation;
            if (!sc::makeGravityCanonicalRotation(query.up, gravity_rotation))
                throw std::runtime_error("invalid query gravity for query " + std::to_string(query.id));

            QueryHypotheses hypotheses;
            if (options.method == "updown" && options.hypotheses_csv.empty())
            {
                hypotheses = makeUpDownHypotheses(source, gravity_rotation, database);
            }
            else
            {
                const auto found = external_hypotheses.find(query.id);
                if (found == external_hypotheses.end())
                    throw std::runtime_error("missing hypotheses for query " + std::to_string(query.id));
                hypotheses = found->second;
            }

            const auto downsample_started = Clock::now();
            const PointCloud::Ptr source_coarse = voxelize(source, 0.5F);
            const PointCloud::Ptr source_fine = voxelize(source, 0.3F);
            const double downsample_ms = std::chrono::duration<double, std::milli>(
                Clock::now() - downsample_started).count();

            for (const int budget : options.budgets)
            {
                std::vector<int> candidate_indices;
                const auto seeds = makeSeeds(
                    options.method, hypotheses.values, budget, database,
                    gravity_rotation, candidate_indices);
                const BudgetResult result = evaluateBudget(
                    icp_config, source_coarse, source_fine, map_coarse, map_fine,
                    seeds, candidate_indices, 5);
                const double icp_ms = result.coarse_ms + result.fine_ms;
                const double total_ms = hypotheses.retrieval_ms + downsample_ms + icp_ms;
                const double dx = result.transform(0, 3) - query.truth_x;
                const double dy = result.transform(1, 3) - query.truth_y;
                const double xy_error = std::hypot(dx, dy);
                const bool truth_success = result.accepted && xy_error <= 2.0;
                output << options.method << ',' << query.id << ',' << query.window << ','
                       << budget << ',' << hypotheses.values.size() << ','
                       << hypotheses.retrieval_ms << ',' << downsample_ms << ','
                       << result.coarse_ms << ',' << result.fine_ms << ',' << icp_ms << ','
                       << total_ms << ',' << result.coarse_converged << ','
                       << result.coarse_valid << ',' << result.fine_converged << ','
                       << result.fine_valid << ',' << (result.accepted ? 1 : 0) << ','
                       << (truth_success ? 1 : 0) << ',' << xy_error << ',';
                if (std::isfinite(result.fitness))
                    output << result.fitness;
                output << ',' << result.overlap << ',' << result.best_seed_rank << ','
                       << result.best_candidate_index << '\n';
            }
            ++evaluated;
            std::cout << options.method << " query " << evaluated << "/92\n";
        }
        std::cout << "Wrote " << options.output_csv << '\n';
    }
    catch (const std::exception &error)
    {
        std::cerr << "offline_icp_budget_evaluator failed: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
