#include "scan_context.hpp"

#include <chrono>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>

namespace sc = fast_lio::scan_context;

namespace
{

bool compatible(const sc::Config &lhs, const sc::Config &rhs)
{
    return lhs.num_rings == rhs.num_rings &&
           lhs.num_sectors == rhs.num_sectors &&
           lhs.max_radius == rhs.max_radius &&
           lhs.dual_z_layer_enable == rhs.dual_z_layer_enable &&
           lhs.gravity_canonicalized == rhs.gravity_canonicalized;
}

void writePose(std::ostream &out, const sc::Pose &pose)
{
    out << pose.x << ',' << pose.y << ',' << pose.z << ','
        << pose.roll << ',' << pose.pitch << ',' << pose.yaw << ','
        << pose.canonical_yaw;
}

bool parsePositiveInt(const char *text, int *value)
{
    if (!text || !value)
        return false;
    errno = 0;
    char *end = nullptr;
    const long parsed = std::strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' ||
        parsed <= 0 || parsed > std::numeric_limits<int>::max())
    {
        return false;
    }
    *value = static_cast<int>(parsed);
    return true;
}

bool parseNonnegativeDouble(const char *text, double *value)
{
    if (!text || !value)
        return false;
    errno = 0;
    char *end = nullptr;
    const double parsed = std::strtod(text, &end);
    if (errno != 0 || end == text || *end != '\0' ||
        !std::isfinite(parsed) || parsed < 0.0)
    {
        return false;
    }
    *value = parsed;
    return true;
}

bool parseBool01(const char *text, bool *value)
{
    if (!text || !value)
        return false;
    const std::string input(text);
    if (input == "0")
    {
        *value = false;
        return true;
    }
    if (input == "1")
    {
        *value = true;
        return true;
    }
    return false;
}

}  // namespace

int main(int argc, char **argv)
{
    if (argc < 4 || argc > 11 || argc == 6 || argc == 7)
    {
        std::cerr
            << "Usage: scan_context_cross_sequence_evaluator "
            << "MAP.scd QUERY.scd OUTPUT.csv [TOP_K "
            << "LOW_WEIGHT HIGH_WEIGHT MIN_JOINT_RINGS "
            << "[SECTOR_SUPPORT_EXPONENT "
            << "[RETRIEVAL_HEIGHT_OFFSET "
            << "[ALLOW_SPLIT_MISMATCH_0_OR_1]]]]\n";
        return 2;
    }

    int top_k = 100;
    if (argc >= 5)
    {
        if (!parsePositiveInt(argv[4], &top_k))
        {
            std::cerr << "TOP_K must be positive\n";
            return 2;
        }
    }
    bool override_scoring = argc >= 8;
    double low_weight = 0.0;
    double high_weight = 0.0;
    int min_joint_rings = 0;
    double sector_support_exponent = sc::Config().sector_support_exponent;
    double retrieval_height_offset = sc::Config().retrieval_height_offset;
    bool allow_split_mismatch = false;
    if (override_scoring &&
        (!parseNonnegativeDouble(argv[5], &low_weight) ||
         !parseNonnegativeDouble(argv[6], &high_weight) ||
         low_weight + high_weight <= 0.0 ||
         !parsePositiveInt(argv[7], &min_joint_rings) ||
         (argc == 9 &&
          !parseNonnegativeDouble(argv[8], &sector_support_exponent)) ||
         (argc == 10 &&
          (!parseNonnegativeDouble(argv[8], &sector_support_exponent) ||
           !parseNonnegativeDouble(argv[9], &retrieval_height_offset))) ||
         (argc == 11 &&
          (!parseNonnegativeDouble(argv[8], &sector_support_exponent) ||
           !parseNonnegativeDouble(argv[9], &retrieval_height_offset) ||
           !parseBool01(argv[10], &allow_split_mismatch)))))
    {
        std::cerr
            << "Weights must be nonnegative with a positive sum, and "
            << "MIN_JOINT_RINGS must be positive\n";
        return 2;
    }

    sc::Config load_config;
    load_config.candidate_top_k = top_k;
    load_config.yaw_top_k = 3;
    load_config.distance_thresh = 1.0;

    sc::Database map_database(load_config);
    sc::Database query_database(load_config);
    std::string error;
    if (!map_database.load(argv[1], &error))
    {
        std::cerr << "Failed to load map database: " << error << '\n';
        return 1;
    }
    if (!query_database.load(argv[2], &error))
    {
        std::cerr << "Failed to load query database: " << error << '\n';
        return 1;
    }
    const bool split_matches =
        std::abs(
            sc::effectiveDualZSplitHeight(map_database.config()) -
            sc::effectiveDualZSplitHeight(query_database.config())) <= 1e-9;
    if (!compatible(map_database.config(), query_database.config()) ||
        (!split_matches && !allow_split_mismatch))
    {
        std::cerr << "Map and query descriptor configurations are incompatible\n";
        return 1;
    }
    if (!split_matches)
    {
        std::cout << "Experimental platform-specific split: map="
                  << sc::effectiveDualZSplitHeight(map_database.config())
                  << " query="
                  << sc::effectiveDualZSplitHeight(query_database.config())
                  << '\n';
    }

    // Loading a database preserves the descriptor geometry from the file and
    // the runtime retrieval budget supplied above.
    sc::Config runtime_config = map_database.config();
    runtime_config.candidate_top_k = top_k;
    runtime_config.yaw_top_k = 3;
    runtime_config.distance_thresh = 1.0;
    if (override_scoring)
    {
        runtime_config.dual_z_low_weight = low_weight;
        runtime_config.dual_z_high_weight = high_weight;
        runtime_config.min_joint_rings = min_joint_rings;
        runtime_config.sector_support_exponent = sector_support_exponent;
        runtime_config.retrieval_height_offset = retrieval_height_offset;
    }
    map_database.setConfig(runtime_config);
    std::cout << "Runtime retrieval scoring: low/high="
              << runtime_config.dual_z_low_weight << '/'
              << runtime_config.dual_z_high_weight
              << ", min_joint_rings=" << runtime_config.min_joint_rings
              << ", sector_support_exponent="
              << runtime_config.sector_support_exponent
              << ", retrieval_height_offset="
              << runtime_config.retrieval_height_offset
              << ", local_absent_upper_fallback_entries="
              << map_database.absentUpperFallbackEntryCount() << '/'
              << map_database.size()
              << ", top_k=" << runtime_config.candidate_top_k << '\n';

    std::ofstream out(argv[3], std::ios::trunc);
    if (!out)
    {
        std::cerr << "Failed to open output CSV: " << argv[3] << '\n';
        return 1;
    }
    out << std::setprecision(17);
    out << "query_index,query_stamp,query_x,query_y,query_z,"
        << "query_roll,query_pitch,query_yaw,query_canonical_yaw,"
        << "candidate_rank,candidate_index,candidate_stamp,"
        << "candidate_x,candidate_y,candidate_z,"
        << "candidate_roll,candidate_pitch,candidate_yaw,candidate_canonical_yaw,"
        << "distance,sector_shift,yaw_shift_rad,"
        << "coarse_vertical_shift,vertical_shift,retrieval_ms\n";
    const std::string timing_path = std::string(argv[3]) + ".timing.csv";
    std::ofstream timing_out(timing_path, std::ios::trunc);
    if (!timing_out)
    {
        std::cerr << "Failed to open timing CSV: " << timing_path << '\n';
        return 1;
    }
    timing_out << std::setprecision(17);
    timing_out << "query_index,retrieval_ms,candidate_count\n";

    const auto &query_entries = query_database.entries();
    const auto &map_entries = map_database.entries();
    for (std::size_t query_index = 0; query_index < query_entries.size(); ++query_index)
    {
        const auto started = std::chrono::steady_clock::now();
        const auto candidates = map_database.queryWithVerticalEstimation(
            query_entries[query_index].descriptor, false);
        const double elapsed_ms =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - started).count();
        timing_out << query_index << ',' << elapsed_ms << ','
                   << candidates.size() << '\n';

        for (std::size_t rank = 0; rank < candidates.size(); ++rank)
        {
            const auto &candidate = candidates[rank];
            if (candidate.index < 0 ||
                candidate.index >= static_cast<int>(map_entries.size()))
            {
                continue;
            }
            const auto &map_entry = map_entries[static_cast<std::size_t>(candidate.index)];
            out << query_index << ',' << query_entries[query_index].stamp << ',';
            writePose(out, query_entries[query_index].pose);
            out << ',' << (rank + 1) << ',' << candidate.index << ','
                << map_entry.stamp << ',';
            writePose(out, map_entry.pose);
            out << ',' << candidate.distance << ',' << candidate.sector_shift << ','
                << candidate.yaw_shift_rad << ','
                << candidate.coarse_vertical_shift << ','
                << candidate.vertical_shift << ',' << elapsed_ms << '\n';
        }

        if ((query_index + 1) % 100 == 0 || query_index + 1 == query_entries.size())
        {
            std::cout << "UpDown-SC query " << (query_index + 1)
                      << '/' << query_entries.size() << '\n';
        }
    }

    if (!out || !timing_out)
    {
        std::cerr << "Failed while writing output or timing CSV\n";
        return 1;
    }
    std::cout << "Wrote cross-sequence candidates to " << argv[3]
              << " (map=" << map_database.size()
              << ", query=" << query_database.size() << ")\n";
    return 0;
}
