#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Geometry>
#include <pcl/io/pcd_io.h>

#include "LidarIris.h"

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

constexpr double kMinRadius = 0.3;
constexpr double kMaxRadius = 30.0;
constexpr int kTopK = 100;

fs::path environmentPath(const char *name, const fs::path &fallback)
{
    const char *value = std::getenv(name);
    return value ? fs::path(value) : fallback;
}

double correctRadius()
{
    static const double value = [] {
        const char *text = std::getenv("IRIS_CORRECT_RADIUS");
        return text ? std::max(0.0, std::stod(text)) : 2.0;
    }();
    return value;
}

bool gravityEnabled()
{
    static const bool value = [] {
        const char *text = std::getenv("IRIS_GRAVITY_CANONICALIZE");
        return text && std::string(text) != "0" && std::string(text) != "false";
    }();
    return value;
}

std::string algorithmName()
{
    return gravityEnabled() ? "LiDAR-Iris + G" : "LiDAR-Iris";
}

const fs::path kMapDir = environmentPath(
    "IRIS_MAP_DIR", "${UPDOWN_SC_ROOT}/icra2027_runtime/manual_loop/gravity/key_point_frame");
const fs::path kMapPoses = environmentPath(
    "IRIS_MAP_POSES", "${UPDOWN_SC_ROOT}/icra2027_runtime/manual_loop/gravity/optimized_poses_tum.txt");
const fs::path kMapGravity = environmentPath(
    "IRIS_MAP_GRAVITY", "${UPDOWN_SC_ROOT}/icra2027_runtime/manual_loop/gravity/scan_context_gravity.csv");
const fs::path kQueryDir = environmentPath(
    "IRIS_QUERY_DIR", "${UPDOWN_SC_ROOT}/icra2027_runtime/experiments/baseline_20260715/queries/loc_2_floor");
const fs::path kOutput = environmentPath(
    "IRIS_OUTPUT", "${UPDOWN_SC_ROOT}/icra2027_runtime/experiments/baseline_20260715/results/lidar_iris_native_per_query.csv");

std::vector<std::string> split(const std::string &line, char delimiter)
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

struct Query
{
    int id = -1;
    int window = -1;
    double start = 0.0;
    bool truth_valid = false;
    double truth_x = 0.0;
    double truth_y = 0.0;
    std::string file;
};

std::vector<Query> loadQueries()
{
    std::ifstream input(kQueryDir / "metadata.csv");
    std::string line;
    std::getline(input, line);
    const auto header = split(line, ',');
    auto column = [&](const std::string &name) {
        const auto found = std::find(header.begin(), header.end(), name);
        if (found == header.end())
            throw std::runtime_error("missing query metadata column: " + name);
        return static_cast<std::size_t>(found - header.begin());
    };
    const auto id = column("query_id"), window = column("window"), start = column("start_s");
    const auto valid = column("truth_valid"), tx = column("truth_x"), ty = column("truth_y");
    const auto file = column("file");
    std::vector<Query> result;
    while (std::getline(input, line))
    {
        const auto fields = split(line, ',');
        Query query;
        query.id = std::stoi(fields[id]);
        query.window = std::stoi(fields[window]);
        query.start = std::stod(fields[start]);
        query.truth_valid = fields[valid] == "True";
        if (query.truth_valid)
        {
            query.truth_x = std::stod(fields[tx]);
            query.truth_y = std::stod(fields[ty]);
        }
        query.file = fields[file];
        result.push_back(query);
    }
    return result;
}

std::vector<Eigen::Vector3d> loadGravity(const fs::path &path)
{
    std::ifstream input(path);
    std::string line;
    std::getline(input, line);
    std::vector<Eigen::Vector3d> result;
    while (std::getline(input, line))
    {
        const auto fields = split(line, ',');
        result.emplace_back(std::stod(fields[2]), std::stod(fields[3]), std::stod(fields[4]));
    }
    return result;
}

std::vector<Eigen::Vector2d> loadMapPositions()
{
    std::ifstream input(kMapPoses);
    std::vector<Eigen::Vector2d> result;
    std::string line;
    while (std::getline(input, line))
    {
        std::stringstream stream(line);
        double stamp, x, y;
        stream >> stamp >> x >> y;
        result.emplace_back(x, y);
    }
    return result;
}

pcl::PointCloud<pcl::PointXYZ> canonicalCrop(
    const pcl::PointCloud<pcl::PointXYZ> &input, const Eigen::Vector3d &up)
{
    Eigen::Quaterniond rotation = Eigen::Quaterniond::Identity();
    if (gravityEnabled())
    {
        if (!up.allFinite() || up.norm() < 1e-12)
            throw std::runtime_error("invalid gravity vector");
        rotation = Eigen::Quaterniond::FromTwoVectors(
            up.normalized(), Eigen::Vector3d::UnitZ());
    }
    pcl::PointCloud<pcl::PointXYZ> output;
    output.reserve(input.size());
    for (const auto &point : input)
    {
        const Eigen::Vector3d transformed =
            rotation * Eigen::Vector3d(point.x, point.y, point.z);
        const double radius = std::hypot(transformed.x(), transformed.y());
        if (!transformed.allFinite() || radius < kMinRadius || radius > kMaxRadius)
            continue;
        output.emplace_back(
            static_cast<float>(transformed.x()),
            static_cast<float>(transformed.y()),
            static_cast<float>(transformed.z()));
    }
    output.width = output.size();
    output.height = 1;
    output.is_dense = true;
    return output;
}

pcl::PointCloud<pcl::PointXYZ> loadQueryCloud(const Query &query)
{
    std::ifstream input(kQueryDir / query.file, std::ios::binary);
    input.seekg(0, std::ios::end);
    const auto bytes = input.tellg();
    input.seekg(0);
    if (bytes % static_cast<std::streamoff>(4 * sizeof(float)) != 0)
        throw std::runtime_error("invalid query binary size");
    std::vector<float> values(static_cast<std::size_t>(bytes) / sizeof(float));
    input.read(reinterpret_cast<char *>(values.data()), bytes);
    pcl::PointCloud<pcl::PointXYZ> cloud;
    cloud.reserve(values.size() / 4);
    for (std::size_t i = 0; i < values.size(); i += 4)
        cloud.emplace_back(values[i], values[i + 1], values[i + 2]);
    return cloud;
}

double percentile(std::vector<double> values, double fraction)
{
    std::sort(values.begin(), values.end());
    const double index = fraction * static_cast<double>(values.size() - 1);
    const auto lower = static_cast<std::size_t>(std::floor(index));
    const auto upper = static_cast<std::size_t>(std::ceil(index));
    const double alpha = index - lower;
    return values[lower] * (1.0 - alpha) + values[upper] * alpha;
}

int main()
{
    try
    {
        LidarIris iris(4, 18, 1.6F, 0.75F, kTopK);
        const auto map_positions = loadMapPositions();
        const auto map_gravity = loadGravity(kMapGravity);
        const auto queries = loadQueries();
        const auto query_gravity = loadGravity(kQueryDir / "gravity.csv");
        if (map_positions.size() != map_gravity.size() || queries.size() != query_gravity.size())
            throw std::runtime_error("pose/gravity count mismatch");

        // File I/O is deliberately outside the retrieval timer.  The timed
        // interval below still includes all descriptor preprocessing, feature
        // construction, key search, and exact Top-K comparison.
        std::vector<pcl::PointCloud<pcl::PointXYZ>> query_clouds(queries.size());
        for (const auto &query : queries)
            if (query.truth_valid)
                query_clouds[query.id] = loadQueryCloud(query);

        const auto map_started = Clock::now();
        std::vector<LidarIris::FeatureDesc> map_features;
        std::vector<std::vector<float>> map_keys;
        map_features.reserve(map_positions.size());
        map_keys.reserve(map_positions.size());
        for (std::size_t i = 0; i < map_positions.size(); ++i)
        {
            pcl::PointCloud<pcl::PointXYZ> cloud;
            if (pcl::io::loadPCDFile((kMapDir / (std::to_string(i) + ".pcd")).string(), cloud) < 0)
                throw std::runtime_error("failed to load map PCD " + std::to_string(i));
            const auto canonical = canonicalCrop(cloud, map_gravity[i]);
            std::vector<float> key;
            map_features.push_back(iris.GetFeature(LidarIris::GetIris(canonical), key));
            map_keys.push_back(std::move(key));
            if ((i + 1) % 250 == 0)
                std::cout << "LiDAR Iris map " << (i + 1) << '/' << map_positions.size() << '\n';
        }
        const double map_build_ms = std::chrono::duration<double, std::milli>(
            Clock::now() - map_started).count();
        const int top_k = std::min<int>(kTopK, map_features.size());
        if (top_k <= 0)
            throw std::runtime_error("empty map feature database");

        std::ofstream output(kOutput);
        output << "algorithm,query_id,window,start_s,truth_x,truth_y,top1_index,top1_x,top1_y,"
                  "top1_error_m,first_correct_rank,recall_at_1,recall_at_5,recall_at_10,"
                  "recall_at_100,retrieval_ms,top1_score\n";
        output << std::setprecision(12);
        int evaluated = 0, r1 = 0, r5 = 0, r10 = 0, r100 = 0;
        std::vector<double> errors, times;
        for (const auto &query : queries)
        {
            if (!query.truth_valid)
                continue;
            const auto started = Clock::now();
            const auto cloud = canonicalCrop(query_clouds[query.id], query_gravity[query.id]);
            std::vector<float> query_key;
            const auto query_feature = iris.GetFeature(LidarIris::GetIris(cloud), query_key);
            std::vector<std::pair<float, int>> key_distances;
            key_distances.reserve(map_keys.size());
            for (std::size_t i = 0; i < map_keys.size(); ++i)
            {
                float distance = 0.0F;
                for (std::size_t j = 0; j < query_key.size(); ++j)
                {
                    const float delta = query_key[j] - map_keys[i][j];
                    distance += delta * delta;
                }
                key_distances.emplace_back(distance, static_cast<int>(i));
            }
            std::partial_sort(
                key_distances.begin(), key_distances.begin() + top_k, key_distances.end());
            std::vector<std::pair<float, int>> exact;
            exact.reserve(top_k);
            for (int i = 0; i < top_k; ++i)
            {
                const int index = key_distances[i].second;
                exact.emplace_back(iris.Compare(query_feature, map_features[index]), index);
            }
            std::sort(exact.begin(), exact.end());
            const double elapsed = std::chrono::duration<double, std::milli>(
                Clock::now() - started).count();
            int correct_rank = 0;
            double top_error = 0.0;
            for (int rank = 0; rank < top_k; ++rank)
            {
                const double error = (map_positions[exact[rank].second] -
                    Eigen::Vector2d(query.truth_x, query.truth_y)).norm();
                if (rank == 0)
                    top_error = error;
                if (correct_rank == 0 && error <= correctRadius())
                    correct_rank = rank + 1;
            }
            const int top_index = exact.front().second;
            ++evaluated;
            r1 += correct_rank > 0 && correct_rank <= 1;
            r5 += correct_rank > 0 && correct_rank <= 5;
            r10 += correct_rank > 0 && correct_rank <= 10;
            r100 += correct_rank > 0 && correct_rank <= 100;
            errors.push_back(top_error);
            times.push_back(elapsed);
            output << algorithmName() << ',' << query.id << ',' << query.window << ',' << query.start << ','
                   << query.truth_x << ',' << query.truth_y << ',' << top_index << ','
                   << map_positions[top_index].x() << ',' << map_positions[top_index].y() << ','
                   << top_error << ',' << (correct_rank ? std::to_string(correct_rank) : "") << ','
                   << (correct_rank == 1) << ',' << (correct_rank > 0 && correct_rank <= 5) << ','
                   << (correct_rank > 0 && correct_rank <= 10) << ','
                   << (correct_rank > 0 && correct_rank <= 100) << ',' << elapsed << ','
                   << exact.front().first << '\n';
            std::cout << "LiDAR Iris query " << evaluated << '/' << queries.size()
                      << " r1=" << r1 << '\n';
        }
        std::cout << "RESULT " << algorithmName() << " n=" << evaluated
                  << " r1=" << static_cast<double>(r1) / evaluated
                  << " r5=" << static_cast<double>(r5) / evaluated
                  << " r10=" << static_cast<double>(r10) / evaluated
                  << " r100=" << static_cast<double>(r100) / evaluated
                  << " error_median=" << percentile(errors, 0.5)
                  << " error_p95=" << percentile(errors, 0.95)
                  << " retrieval_ms_median=" << percentile(times, 0.5)
                  << " retrieval_ms_p95=" << percentile(times, 0.95)
                  << " map_build_ms=" << map_build_ms << '\n';
    }
    catch (const std::exception &error)
    {
        std::cerr << "LiDAR Iris benchmark failed: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
