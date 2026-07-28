#include "scan_context.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <Eigen/Geometry>
#include <sstream>
#include <stdexcept>
#include <system_error>

namespace fast_lio
{
namespace scan_context
{
namespace
{

constexpr const char *kDatabaseMagicV1 = "FAST_LIO_SCAN_CONTEXT_DB_V1";
constexpr const char *kDatabaseMagicV2 = "FAST_LIO_SCAN_CONTEXT_DB_V2";
constexpr const char *kDatabaseMagicV3 = "FAST_LIO_SCAN_CONTEXT_DB_V3";
constexpr const char *kDatabaseMagicV4 = "FAST_LIO_SCAN_CONTEXT_DB_V4";
constexpr const char *kDatabaseMagicV5 = "FAST_LIO_SCAN_CONTEXT_DB_V5";
constexpr const char *kDatabaseMagicV6 = "FAST_LIO_SCAN_CONTEXT_DB_V6";
constexpr const char *kDatabaseMagicV7 = "FAST_LIO_SCAN_CONTEXT_DB_V7";
constexpr double kTwoPi = 2.0 * M_PI;
constexpr int kMaxRings = 512;
constexpr int kMaxSectors = 4096;
constexpr std::size_t kMaxDescriptorCells = 1000000;
constexpr std::size_t kMaxDatabaseEntries = 100000;

double normalizeAngle(double angle)
{
    if (!std::isfinite(angle))
        return 0.0;
    angle = std::remainder(angle, kTwoPi);
    if (angle <= -M_PI)
        angle += kTwoPi;
    return angle;
}

bool validConfig(const Config &config)
{
    const int descriptor_rows = config.dual_z_layer_enable ? config.num_rings * 2 : config.num_rings;
    const std::size_t descriptor_cells =
        descriptor_rows > 0 && config.num_sectors > 0
            ? static_cast<std::size_t>(descriptor_rows) * static_cast<std::size_t>(config.num_sectors)
            : 0;
    return config.num_rings > 0 &&
           config.num_rings <= kMaxRings &&
           config.num_sectors > 0 &&
           config.num_sectors <= kMaxSectors &&
           descriptor_cells <= kMaxDescriptorCells &&
           std::isfinite(config.max_radius) &&
           config.max_radius > 1e-6 &&
           std::isfinite(config.dual_z_split_height) &&
           std::isfinite(config.dual_z_split_auto_min) &&
           std::isfinite(config.dual_z_split_auto_max) &&
           config.dual_z_split_auto_min >= 0.0 &&
           config.dual_z_split_auto_min < config.dual_z_split_auto_max &&
           std::isfinite(config.dual_z_split_auto_bin_size) &&
           config.dual_z_split_auto_bin_size > 1e-6 &&
           std::isfinite(config.dual_z_split_auto_histogram_max) &&
           config.dual_z_split_auto_histogram_max >
               config.dual_z_split_auto_max &&
           std::isfinite(config.dual_z_split_auto_min_layer_fraction) &&
           config.dual_z_split_auto_min_layer_fraction > 0.0 &&
           config.dual_z_split_auto_min_layer_fraction < 0.5 &&
           config.dual_z_split_auto_min_keyframes > 0 &&
           std::isfinite(config.origin_height_from_ground) &&
           config.origin_height_from_ground >= 0.0 &&
           std::isfinite(config.dual_z_low_weight) &&
           std::isfinite(config.dual_z_high_weight) &&
           config.dual_z_low_weight >= 0.0 &&
           config.dual_z_high_weight >= 0.0 &&
           (config.dual_z_low_weight + config.dual_z_high_weight) > 1e-12 &&
           config.min_joint_rings > 0 &&
           config.min_joint_rings <= config.num_rings &&
           std::isfinite(config.absent_upper_fallback_max_local_fraction) &&
           config.absent_upper_fallback_max_local_fraction >= 0.0 &&
           config.absent_upper_fallback_max_local_fraction <= 1.0 &&
           std::isfinite(config.absent_upper_fallback_radius) &&
           config.absent_upper_fallback_radius > 0.0 &&
           config.absent_upper_fallback_min_keyframes > 0 &&
           std::isfinite(config.retrieval_height_offset) &&
           std::isfinite(config.sector_support_exponent) &&
           config.sector_support_exponent >= 0.0 &&
           std::isfinite(config.vertical_boundary_margin) &&
           config.vertical_boundary_margin >= 0.0 &&
           std::isfinite(config.vertical_correction_min) &&
           std::isfinite(config.vertical_correction_max) &&
           config.vertical_correction_min <= config.vertical_correction_max &&
           std::isfinite(config.vertical_stable_fraction) &&
           config.vertical_stable_fraction > 0.0 &&
           config.vertical_stable_fraction <= 1.0 &&
           config.candidate_top_k > 0 &&
           config.yaw_top_k > 0 &&
           std::isfinite(config.distance_thresh) &&
           config.distance_thresh > 0.0;
}

int descriptorRows(const Config &config)
{
    return config.dual_z_layer_enable ? config.num_rings * 2 : config.num_rings;
}

bool finitePose(double stamp, const Pose &pose)
{
    return std::isfinite(stamp) &&
           std::isfinite(pose.x) &&
           std::isfinite(pose.y) &&
           std::isfinite(pose.z) &&
           std::isfinite(pose.roll) &&
           std::isfinite(pose.pitch) &&
           std::isfinite(pose.yaw) &&
           std::isfinite(pose.canonical_yaw);
}

std::vector<std::uint8_t> packValidityMask(const Descriptor &descriptor)
{
    const std::size_t rows = static_cast<std::size_t>(descriptor.valid.rows());
    const std::size_t cols = static_cast<std::size_t>(descriptor.valid.cols());
    const std::size_t cell_count = rows * cols;
    std::vector<std::uint8_t> packed((cell_count + 7U) / 8U, 0U);
    for (std::size_t row = 0; row < rows; ++row)
    {
        for (std::size_t col = 0; col < cols; ++col)
        {
            const std::size_t bit_index = row * cols + col;
            if (descriptor.valid(static_cast<int>(row), static_cast<int>(col)) != 0U)
                packed[bit_index / 8U] |= static_cast<std::uint8_t>(1U << (bit_index % 8U));
        }
    }
    return packed;
}

bool readRequiredLineBreak(std::istream &in)
{
    char separator = '\0';
    if (!in.get(separator))
        return false;
    if (separator == '\n')
        return true;
    if (separator != '\r')
        return false;
    return static_cast<bool>(in.get(separator)) && separator == '\n';
}

}  // namespace

double effectiveDualZSplitHeight(const Config &config)
{
    return config.dual_z_split_height;
}

AdaptiveSplitEstimator::AdaptiveSplitEstimator(const Config &config)
{
    setConfig(config);
}

void AdaptiveSplitEstimator::setConfig(const Config &config)
{
    if (!validConfig(config))
        throw std::invalid_argument("invalid Scan Context configuration");
    config_ = config;
    clear();
}

void AdaptiveSplitEstimator::clear()
{
    const std::size_t bin_count = static_cast<std::size_t>(
        std::ceil(
            config_.dual_z_split_auto_histogram_max /
            config_.dual_z_split_auto_bin_size));
    histogram_.assign(std::max<std::size_t>(1, bin_count), 0);
    keyframe_count_ = 0;
}

void AdaptiveSplitEstimator::addScan(const PointCloud &scan)
{
    if (histogram_.empty())
        return;

    const std::size_t cell_count =
        static_cast<std::size_t>(config_.num_rings) *
        static_cast<std::size_t>(config_.num_sectors);
    std::vector<std::uint8_t> occupied(
        cell_count * histogram_.size(), 0);
    bool have_support = false;
    for (const auto &point : scan.points)
    {
        if (!std::isfinite(point.x) || !std::isfinite(point.y) ||
            !std::isfinite(point.z))
        {
            continue;
        }
        const double range =
            std::hypot(static_cast<double>(point.x), static_cast<double>(point.y));
        if (range <= 1e-6 || range > config_.max_radius)
            continue;
        const double height =
            static_cast<double>(point.z) + config_.origin_height_from_ground;
        if (height < 0.0 ||
            height >= config_.dual_z_split_auto_histogram_max)
        {
            continue;
        }

        double theta =
            std::atan2(static_cast<double>(point.y), static_cast<double>(point.x));
        if (theta < 0.0)
            theta += kTwoPi;
        const int ring = std::clamp(
            static_cast<int>(
                std::ceil(
                    range / config_.max_radius * config_.num_rings)) -
                1,
            0, config_.num_rings - 1);
        const int sector = std::clamp(
            static_cast<int>(
                std::ceil(
                    theta / kTwoPi * config_.num_sectors)) -
                1,
            0, config_.num_sectors - 1);
        const std::size_t bin = std::min(
            histogram_.size() - 1,
            static_cast<std::size_t>(
                height / config_.dual_z_split_auto_bin_size));
        const std::size_t cell =
            static_cast<std::size_t>(ring) *
                static_cast<std::size_t>(config_.num_sectors) +
            static_cast<std::size_t>(sector);
        occupied[cell * histogram_.size() + bin] = 1;
        have_support = true;
    }

    if (!have_support)
        return;
    for (std::size_t cell = 0; cell < cell_count; ++cell)
    {
        const std::size_t offset = cell * histogram_.size();
        for (std::size_t bin = 0; bin < histogram_.size(); ++bin)
            histogram_[bin] += occupied[offset + bin] != 0 ? 1 : 0;
    }
    ++keyframe_count_;
}

AdaptiveSplitResult AdaptiveSplitEstimator::estimate() const
{
    AdaptiveSplitResult result;
    result.dual_layer_enabled = config_.dual_z_layer_enable;
    result.split_height = config_.dual_z_split_height;
    result.keyframe_count = keyframe_count_;
    result.support_count = std::accumulate(
        histogram_.begin(), histogram_.end(), std::uint64_t{0});
    if (!config_.dual_z_layer_enable || !config_.dual_z_split_auto)
        return result;
    if (keyframe_count_ <
            static_cast<std::size_t>(
                config_.dual_z_split_auto_min_keyframes) ||
        result.support_count == 0)
    {
        result.adapted = true;
        result.dual_layer_enabled = false;
        return result;
    }

    long double total_height = 0.0;
    for (std::size_t bin = 0; bin < histogram_.size(); ++bin)
    {
        const double center =
            (static_cast<double>(bin) + 0.5) *
            config_.dual_z_split_auto_bin_size;
        total_height +=
            static_cast<long double>(histogram_[bin]) * center;
    }

    std::uint64_t lower_count = 0;
    long double lower_height = 0.0;
    double best_score = -1.0;
    std::uint64_t best_lower_count = 0;
    double best_threshold_min = 0.0;
    double best_threshold_max = 0.0;
    for (std::size_t bin = 0; bin + 1 < histogram_.size(); ++bin)
    {
        const double center =
            (static_cast<double>(bin) + 0.5) *
            config_.dual_z_split_auto_bin_size;
        lower_count += histogram_[bin];
        lower_height +=
            static_cast<long double>(histogram_[bin]) * center;
        const double threshold =
            (static_cast<double>(bin) + 1.0) *
            config_.dual_z_split_auto_bin_size;
        if (threshold < config_.dual_z_split_auto_min ||
            threshold > config_.dual_z_split_auto_max)
        {
            continue;
        }
        const std::uint64_t upper_count =
            result.support_count - lower_count;
        const double lower_fraction =
            static_cast<double>(lower_count) /
            static_cast<double>(result.support_count);
        const double upper_fraction =
            static_cast<double>(upper_count) /
            static_cast<double>(result.support_count);
        if (lower_fraction <
                config_.dual_z_split_auto_min_layer_fraction ||
            upper_fraction <
                config_.dual_z_split_auto_min_layer_fraction)
        {
            continue;
        }
        const long double upper_height = total_height - lower_height;
        const double lower_mean =
            static_cast<double>(
                lower_height / static_cast<long double>(lower_count));
        const double upper_mean =
            static_cast<double>(
                upper_height / static_cast<long double>(upper_count));
        const double mean_gap = lower_mean - upper_mean;
        const double score =
            lower_fraction * upper_fraction * mean_gap * mean_gap;
        const bool equal_score =
            std::abs(score - best_score) <=
            1e-12 * std::max(1.0, std::abs(best_score));
        if (score > best_score && !equal_score)
        {
            best_score = score;
            best_threshold_min = threshold;
            best_threshold_max = threshold;
            best_lower_count = lower_count;
        }
        else if (equal_score)
        {
            best_threshold_max = threshold;
        }
    }
    if (best_score < 0.0)
    {
        result.adapted = true;
        result.dual_layer_enabled = false;
        return result;
    }

    result.split_height =
        0.5 * (best_threshold_min + best_threshold_max);
    result.lower_fraction =
        static_cast<double>(best_lower_count) /
        static_cast<double>(result.support_count);
    result.upper_fraction = 1.0 - result.lower_fraction;
    result.between_class_variance = best_score;
    result.adapted = true;
    return result;
}

Database::Database(const Config &config)
{
    setConfig(config);
}

const Config &Database::config() const
{
    return config_;
}

void Database::setConfig(const Config &config)
{
    config_ = config;
    if (!validConfig(config_))
    {
        config_ = Config();
    }
    for (auto &entry : entries_)
    {
        refreshKeys(entry);
    }
    entries_.erase(
        std::remove_if(entries_.begin(), entries_.end(), [this](const Entry &entry) {
            return entry.ring_key.size() != descriptorRows(config_) ||
                   entry.sector_key.size() != config_.num_sectors;
        }),
        entries_.end());
    observation_stats_dirty_ = true;
}

void Database::clear()
{
    entries_.clear();
    legacy_masks_inferred_ = false;
    entries_with_upper_observation_ = 0;
    entry_allows_absent_upper_fallback_.clear();
    observation_stats_dirty_ = false;
}

bool Database::empty() const
{
    return entries_.empty();
}

std::size_t Database::size() const
{
    return entries_.size();
}

const std::vector<Entry> &Database::entries() const
{
    return entries_;
}

bool Database::legacyMasksInferred() const
{
    return legacy_masks_inferred_;
}

std::size_t Database::absentUpperFallbackEntryCount() const
{
    ensureObservationStats();
    return static_cast<std::size_t>(std::count(
        entry_allows_absent_upper_fallback_.begin(),
        entry_allows_absent_upper_fallback_.end(),
        static_cast<std::uint8_t>(1)));
}

Descriptor Database::makeDescriptor(const PointCloud &scan,
                                    double vertical_shift) const
{
    Descriptor descriptor;
    descriptor.values = Eigen::MatrixXd::Zero(descriptorRows(config_), config_.num_sectors);
    descriptor.valid = Eigen::Array<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic>::Zero(
        descriptorRows(config_), config_.num_sectors);

    for (const auto &point : scan.points)
    {
        if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z))
            continue;

        const double range = std::hypot(static_cast<double>(point.x), static_cast<double>(point.y));
        if (range <= 1e-6 || range > config_.max_radius)
            continue;

        double theta = std::atan2(static_cast<double>(point.y), static_cast<double>(point.x));
        if (theta < 0.0)
            theta += kTwoPi;

        const int ring_idx = std::max(
            0,
            std::min(config_.num_rings - 1,
                     static_cast<int>(std::ceil((range / config_.max_radius) * config_.num_rings)) - 1));
        const int sector_idx = std::max(
            0,
            std::min(config_.num_sectors - 1,
                     static_cast<int>(std::ceil((theta / kTwoPi) * config_.num_sectors)) - 1));

        const double shifted_z = static_cast<double>(point.z) + vertical_shift;
        const double height =
            shifted_z + config_.origin_height_from_ground;
        if (!config_.dual_z_layer_enable ||
            height <= effectiveDualZSplitHeight(config_))
        {
            if (!descriptor.valid(ring_idx, sector_idx) ||
                descriptor.values(ring_idx, sector_idx) < height)
            {
                descriptor.values(ring_idx, sector_idx) = height;
            }
            descriptor.valid(ring_idx, sector_idx) = 1;
        }
        else
        {
            const int high_ring_idx = config_.num_rings + ring_idx;
            if (!descriptor.valid(high_ring_idx, sector_idx) ||
                descriptor.values(high_ring_idx, sector_idx) > height)
            {
                descriptor.values(high_ring_idx, sector_idx) = height;
            }
            descriptor.valid(high_ring_idx, sector_idx) = 1;
        }
    }

    descriptor.boundary = makeBoundaryMask(descriptor);

    return descriptor;
}

void Database::addEntry(double stamp, const Pose &pose, const Descriptor &descriptor)
{
    if (!finitePose(stamp, pose) || !validateDescriptorShape(descriptor))
        return;

    Entry entry;
    entry.stamp = stamp;
    entry.pose = pose;
    if (!config_.gravity_canonicalized)
        entry.pose.canonical_yaw = entry.pose.yaw;
    entry.descriptor = descriptor;
    refreshKeys(entry);
    entries_.push_back(entry);
    observation_stats_dirty_ = true;
}

std::vector<Candidate> Database::query(const Descriptor &descriptor,
                                       bool apply_distance_threshold) const
{
    std::vector<Candidate> candidates;
    if (entries_.empty() || !validateDescriptorShape(descriptor))
        return candidates;

    const Eigen::VectorXd query_ring_key = makeRingKey(descriptor);
    std::vector<std::pair<double, int>> ring_distances;
    ring_distances.reserve(entries_.size());
    for (int i = 0; i < static_cast<int>(entries_.size()); ++i)
    {
        if (entries_[i].ring_key.size() != query_ring_key.size())
            continue;
        const double ring_distance = (query_ring_key - entries_[i].ring_key).norm();
        if (!std::isfinite(ring_distance))
            continue;
        ring_distances.emplace_back(ring_distance, i);
    }
    if (ring_distances.empty())
        return candidates;

    std::sort(ring_distances.begin(), ring_distances.end(),
              [](const auto &lhs, const auto &rhs) {
                  return lhs.first < rhs.first;
              });

    const int top_k = std::max(
        1,
        std::min(config_.candidate_top_k, static_cast<int>(ring_distances.size())));
    candidates.reserve(top_k);
    ensureObservationStats();

    for (int i = 0; i < top_k; ++i)
    {
        const int entry_idx = ring_distances[i].second;
        const bool allow_absent_upper =
            entry_idx >= 0 &&
            entry_idx <
                static_cast<int>(
                    entry_allows_absent_upper_fallback_.size()) &&
            entry_allows_absent_upper_fallback_[entry_idx] != 0;
        const auto yaw_matches = yawMatchesImpl(
            descriptor, entries_[entry_idx].descriptor, config_.yaw_top_k,
            allow_absent_upper);
        if (yaw_matches.empty())
            continue;
        if (apply_distance_threshold && yaw_matches.front().distance > config_.distance_thresh)
            continue;

        Candidate candidate;
        candidate.index = entry_idx;
        candidate.distance = yaw_matches.front().distance;
        candidate.sector_shift = yaw_matches.front().sector_shift;
        candidate.yaw_shift_rad = yaw_matches.front().yaw_shift_rad;
        candidate.coarse_vertical_shift = yaw_matches.front().coarse_vertical_shift;
        candidate.vertical_shift = yaw_matches.front().vertical_shift;
        candidate.pose = entries_[entry_idx].pose;
        candidate.yaw_matches = yaw_matches;
        candidates.push_back(candidate);
    }

    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate &lhs, const Candidate &rhs) {
                  return lhs.distance < rhs.distance;
              });
    return candidates;
}

std::vector<Candidate> Database::query(const PointCloud &scan,
                                       bool apply_distance_threshold) const
{
    const Descriptor zero_descriptor = makeDescriptor(scan, 0.0);
    return queryWithVerticalEstimation(
        zero_descriptor, apply_distance_threshold);
}

std::vector<Candidate> Database::queryWithVerticalEstimation(
    const Descriptor &zero_descriptor,
    bool apply_distance_threshold) const
{
    std::vector<Candidate> candidates =
        query(zero_descriptor, apply_distance_threshold);
    if (!config_.dual_z_layer_enable || !config_.vertical_estimation_enable)
        return candidates;

    auto bounded_vertical_shift = [this](double value, double fallback) {
        if (!std::isfinite(value) ||
            value < config_.vertical_correction_min ||
            value > config_.vertical_correction_max)
        {
            return fallback;
        }
        return value;
    };

    for (auto &candidate : candidates)
    {
        for (auto &match : candidate.yaw_matches)
        {
            // The map candidate and yaw come exclusively from the legacy h=0
            // Scan Context match. Height is estimated afterwards and therefore
            // cannot change candidate ranking or evict a yaw hypothesis.
            const double vertical_shift = bounded_vertical_shift(
                estimateVerticalShift(
                    zero_descriptor,
                    entries_[candidate.index].descriptor,
                    match.sector_shift),
                0.0);
            // Retain both fields for CSV/log compatibility. This method is now
            // deliberately one-pass: it never rebuilds a split-dependent mask.
            match.coarse_vertical_shift = vertical_shift;
            match.vertical_shift = vertical_shift;
        }
        if (!candidate.yaw_matches.empty())
        {
            candidate.coarse_vertical_shift =
                candidate.yaw_matches.front().coarse_vertical_shift;
            candidate.vertical_shift = candidate.yaw_matches.front().vertical_shift;
        }
    }
    return candidates;
}

bool Database::save(const std::string &path, std::string *error) const
{
    auto fail = [error](const std::string &message) {
        if (error)
            *error = message;
        return false;
    };

    if (path.empty())
        return fail("empty Scan Context database path");

    const std::filesystem::path db_path(path);
    if (!db_path.parent_path().empty())
    {
        std::error_code ec;
        std::filesystem::create_directories(db_path.parent_path(), ec);
        if (ec)
            return fail("failed to create directory: " + ec.message());
    }

    const std::filesystem::path tmp_path = db_path.string() + ".tmp";
    auto cleanup_tmp = [&tmp_path]() {
        std::error_code cleanup_ec;
        std::filesystem::remove(tmp_path, cleanup_ec);
    };

    {
        std::ofstream out(tmp_path, std::ios::binary | std::ios::trunc);
        if (!out)
        {
            cleanup_tmp();
            return fail("failed to open temporary Scan Context database for writing");
        }

        out << std::setprecision(17);
        out << kDatabaseMagicV7 << '\n';
        out << "PARAMS "
            << config_.num_rings << ' '
            << config_.num_sectors << ' '
            << config_.max_radius << ' '
            << (config_.dual_z_layer_enable ? 1 : 0) << ' '
            << config_.dual_z_split_height << ' '
            << config_.origin_height_from_ground << ' '
            << config_.dual_z_low_weight << ' '
            << config_.dual_z_high_weight << ' '
            << config_.min_joint_rings << ' '
            << (config_.gravity_canonicalized ? 1 : 0) << '\n';
        out << "ENTRIES " << entries_.size() << '\n';

        for (int i = 0; i < static_cast<int>(entries_.size()); ++i)
        {
            const auto &entry = entries_[i];
            out << "ENTRY " << i << ' '
                << entry.stamp << ' '
                << entry.pose.x << ' '
                << entry.pose.y << ' '
                << entry.pose.z << ' '
                << entry.pose.roll << ' '
                << entry.pose.pitch << ' '
                << entry.pose.yaw << ' '
                << entry.pose.canonical_yaw << '\n';
            out << "DESC\n";
            for (int row = 0; row < entry.descriptor.values.rows(); ++row)
            {
                for (int col = 0; col < entry.descriptor.values.cols(); ++col)
                {
                    if (col > 0)
                        out << ' ';
                    out << entry.descriptor.values(row, col);
                }
                out << '\n';
            }
            const std::vector<std::uint8_t> packed_mask = packValidityMask(entry.descriptor);
            out << "MASK_BITS " << packed_mask.size() << '\n';
            out.write(
                reinterpret_cast<const char *>(packed_mask.data()),
                static_cast<std::streamsize>(packed_mask.size()));
            out << '\n';
            out << "END_ENTRY\n";
        }

        out.flush();
        if (!out)
        {
            cleanup_tmp();
            return fail("failed while writing Scan Context database");
        }
    }

    std::error_code rename_ec;
    std::filesystem::rename(tmp_path, db_path, rename_ec);
    if (rename_ec)
    {
        std::error_code remove_ec;
        std::filesystem::remove(db_path, remove_ec);
        rename_ec.clear();
        std::filesystem::rename(tmp_path, db_path, rename_ec);
        if (rename_ec)
        {
            cleanup_tmp();
            return fail("failed to move temporary Scan Context database into place: " + rename_ec.message());
        }
    }

    return true;
}

bool Database::load(const std::string &path, std::string *error)
{
    auto fail = [error](const std::string &message) {
        if (error)
            *error = message;
        return false;
    };

    std::ifstream in(path, std::ios::binary);
    if (!in)
        return fail("failed to open Scan Context database");

    std::string token;
    if (!(in >> token))
        return fail("failed to read Scan Context database magic");
    const bool is_v1 = token == kDatabaseMagicV1;
    const bool is_v2 = token == kDatabaseMagicV2;
    const bool is_v3 = token == kDatabaseMagicV3;
    const bool is_v4 = token == kDatabaseMagicV4;
    const bool is_v5 = token == kDatabaseMagicV5;
    const bool is_v6 = token == kDatabaseMagicV6;
    const bool is_v7 = token == kDatabaseMagicV7;
    if (!is_v1 && !is_v2 && !is_v3 && !is_v4 && !is_v5 && !is_v6 && !is_v7)
        return fail("invalid Scan Context database magic");

    Config loaded_config = config_;
    if (!(in >> token))
        return fail("missing PARAMS section");
    if (token != "PARAMS")
        return fail("missing PARAMS section");
    std::string params_line;
    std::getline(in, params_line);
    if (params_line.empty())
        std::getline(in, params_line);
    std::istringstream params_stream(params_line);
    if (!(params_stream >> loaded_config.num_rings
                        >> loaded_config.num_sectors
                        >> loaded_config.max_radius))
    {
        return fail("failed to read Scan Context parameters");
    }
    double legacy_height_offset = 0.0;
    if (!is_v5 && !is_v6 && !is_v7 &&
        (!(params_stream >> legacy_height_offset) || !std::isfinite(legacy_height_offset)))
    {
        return fail("failed to read legacy Scan Context height offset");
    }
    int dual_z_layer_enable = 0;
    if (params_stream >> dual_z_layer_enable
                      >> loaded_config.dual_z_split_height)
    {
        if (is_v6 || is_v7)
        {
            if (!(params_stream >> loaded_config.origin_height_from_ground))
                return fail("failed to read Scan Context origin height");
        }
        else
        {
            // V1--V5 do not contain platform-origin metadata. Treat their
            // stored frame as ground-referenced for compatibility.
            loaded_config.origin_height_from_ground = 0.0;
        }
        if (!(params_stream >> loaded_config.dual_z_low_weight
                            >> loaded_config.dual_z_high_weight))
        {
            return fail("failed to read Scan Context dual-layer weights");
        }
        loaded_config.dual_z_layer_enable = dual_z_layer_enable != 0;
    }
    else
    {
        loaded_config.dual_z_layer_enable = false;
        loaded_config.dual_z_split_height = Config().dual_z_split_height;
        loaded_config.origin_height_from_ground = 0.0;
        loaded_config.dual_z_low_weight = Config().dual_z_low_weight;
        loaded_config.dual_z_high_weight = Config().dual_z_high_weight;
    }
    if ((is_v2 || is_v3 || is_v4 || is_v5 || is_v6 || is_v7) &&
        !(params_stream >> loaded_config.min_joint_rings))
        return fail("failed to read Scan Context joint-support parameter");
    if (is_v4 || is_v5 || is_v6 || is_v7)
    {
        int gravity_canonicalized = 0;
        if (!(params_stream >> gravity_canonicalized) ||
            (gravity_canonicalized != 0 && gravity_canonicalized != 1))
        {
            return fail("failed to read Scan Context gravity-canonicalization parameter");
        }
        loaded_config.gravity_canonicalized = gravity_canonicalized != 0;
    }
    else
    {
        loaded_config.gravity_canonicalized = false;
    }
    loaded_config.candidate_top_k = config_.candidate_top_k;
    loaded_config.yaw_top_k = config_.yaw_top_k;
    loaded_config.distance_thresh = config_.distance_thresh;
    if (!validConfig(loaded_config))
        return fail("invalid Scan Context parameters");

    std::size_t entry_count = 0;
    if (!(in >> token))
        return fail("missing ENTRIES section");
    if (token != "ENTRIES")
        return fail("missing ENTRIES section");
    if (!(in >> entry_count))
        return fail("failed to read Scan Context entry count");
    if (entry_count > kMaxDatabaseEntries)
        return fail("Scan Context entry count is too large");

    std::vector<Entry> loaded_entries;
    loaded_entries.reserve(entry_count);
    for (std::size_t entry_idx = 0; entry_idx < entry_count; ++entry_idx)
    {
        if (!(in >> token))
            return fail("missing ENTRY section");
        if (token != "ENTRY")
            return fail("missing ENTRY section");

        int file_index = -1;
        Entry entry;
        if (!(in >> file_index
                 >> entry.stamp
                 >> entry.pose.x
                 >> entry.pose.y
                 >> entry.pose.z
                 >> entry.pose.roll
                 >> entry.pose.pitch
                 >> entry.pose.yaw))
        {
            return fail("failed to read Scan Context entry pose");
        }
        if (is_v4 || is_v5 || is_v6 || is_v7)
        {
            if (!(in >> entry.pose.canonical_yaw))
                return fail("failed to read Scan Context canonical yaw");
        }
        else
        {
            entry.pose.canonical_yaw = entry.pose.yaw;
        }
        if (!finitePose(entry.stamp, entry.pose))
            return fail("Scan Context entry pose contains non-finite value");
        (void)file_index;

        if (!(in >> token))
            return fail("missing DESC section");
        if (token != "DESC")
            return fail("missing DESC section");

        const int loaded_descriptor_rows = descriptorRows(loaded_config);
        entry.descriptor.values.resize(loaded_descriptor_rows, loaded_config.num_sectors);
        entry.descriptor.valid = Eigen::Array<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic>::Zero(
            loaded_descriptor_rows, loaded_config.num_sectors);
        for (int row = 0; row < loaded_descriptor_rows; ++row)
        {
            for (int col = 0; col < loaded_config.num_sectors; ++col)
            {
                if (!(in >> entry.descriptor.values(row, col)))
                    return fail("failed to read Scan Context descriptor values");
                if (!std::isfinite(entry.descriptor.values(row, col)))
                    return fail("Scan Context descriptor contains non-finite value");
            }
        }

        if (is_v2)
        {
            if (!(in >> token) || token != "MASK")
                return fail("missing MASK section");
            for (int row = 0; row < loaded_descriptor_rows; ++row)
            {
                for (int col = 0; col < loaded_config.num_sectors; ++col)
                {
                    int valid = 0;
                    if (!(in >> valid) || (valid != 0 && valid != 1))
                        return fail("failed to read Scan Context validity mask");
                    entry.descriptor.valid(row, col) = static_cast<std::uint8_t>(valid);
                }
            }
        }
        else if (is_v3 || is_v4 || is_v5 || is_v6 || is_v7)
        {
            if (!(in >> token) || token != "MASK_BITS")
                return fail("missing MASK_BITS section");
            std::size_t packed_size = 0;
            if (!(in >> packed_size))
                return fail("failed to read Scan Context bitset mask size");
            const std::size_t cell_count =
                static_cast<std::size_t>(loaded_descriptor_rows) *
                static_cast<std::size_t>(loaded_config.num_sectors);
            const std::size_t expected_size = (cell_count + 7U) / 8U;
            if (packed_size != expected_size)
                return fail("Scan Context bitset mask size does not match descriptor shape");
            if (!readRequiredLineBreak(in))
                return fail("missing line break before Scan Context bitset mask data");

            std::vector<std::uint8_t> packed_mask(packed_size, 0U);
            in.read(
                reinterpret_cast<char *>(packed_mask.data()),
                static_cast<std::streamsize>(packed_mask.size()));
            if (in.gcount() != static_cast<std::streamsize>(packed_mask.size()))
                return fail("failed to read Scan Context bitset mask data");
            if (!readRequiredLineBreak(in))
                return fail("missing line break after Scan Context bitset mask data");

            for (std::size_t bit_index = 0; bit_index < cell_count; ++bit_index)
            {
                const int row = static_cast<int>(bit_index / static_cast<std::size_t>(loaded_config.num_sectors));
                const int col = static_cast<int>(bit_index % static_cast<std::size_t>(loaded_config.num_sectors));
                entry.descriptor.valid(row, col) = static_cast<std::uint8_t>(
                    (packed_mask[bit_index / 8U] >> (bit_index % 8U)) & 1U);
            }
        }
        else
        {
            entry.descriptor.valid =
                (entry.descriptor.values.array() != 0.0).cast<std::uint8_t>();
        }

        // V1--V4 stored z + height_offset. First recover their signed height.
        if (!is_v5 && !is_v6 && !is_v7 && legacy_height_offset != 0.0)
        {
            for (int row = 0; row < entry.descriptor.values.rows(); ++row)
            {
                for (int col = 0; col < entry.descriptor.values.cols(); ++col)
                {
                    if (entry.descriptor.valid(row, col))
                        entry.descriptor.values(row, col) -= legacy_height_offset;
                }
            }
        }

        // V6 stored signed descriptor-origin heights while also recording the
        // platform origin height. Convert valid cells to the V7 ground-relative
        // representation during loading. V1--V5 lack reliable platform-height
        // metadata and therefore remain numerically unchanged.
        if (is_v6 && loaded_config.origin_height_from_ground != 0.0)
        {
            for (int row = 0; row < entry.descriptor.values.rows(); ++row)
            {
                for (int col = 0; col < entry.descriptor.values.cols(); ++col)
                {
                    if (entry.descriptor.valid(row, col))
                    {
                        entry.descriptor.values(row, col) +=
                            loaded_config.origin_height_from_ground;
                    }
                }
            }
        }

        if (!(in >> token))
            return fail("missing END_ENTRY marker");
        if (token != "END_ENTRY")
            return fail("missing END_ENTRY marker");

        loaded_entries.push_back(entry);
    }

    if (!in.good() && !in.eof())
        return fail("failed while reading Scan Context database");

    config_ = loaded_config;
    entries_ = std::move(loaded_entries);
    legacy_masks_inferred_ = is_v1;
    for (auto &entry : entries_)
    {
        refreshKeys(entry);
    }
    observation_stats_dirty_ = true;
    ensureObservationStats();
    return true;
}

double Database::distance(const Descriptor &query,
                          const Descriptor &candidate,
                          int *sector_shift) const
{
    const auto matches = yawMatches(query, candidate, 1);
    if (matches.empty())
    {
        if (sector_shift)
            *sector_shift = 0;
        return std::numeric_limits<double>::infinity();
    }

    if (sector_shift)
        *sector_shift = matches.front().sector_shift;
    return matches.front().distance;
}

std::vector<YawMatch> Database::yawMatches(const Descriptor &query,
                                           const Descriptor &candidate,
                                           int max_matches) const
{
    ensureObservationStats();
    const double global_upper_fraction = entries_.empty()
        ? 0.0
        : static_cast<double>(entries_with_upper_observation_) /
              static_cast<double>(entries_.size());
    return yawMatchesImpl(
        query, candidate, max_matches,
        entries_.empty() ||
            global_upper_fraction <=
                config_.absent_upper_fallback_max_local_fraction);
}

std::vector<YawMatch> Database::yawMatchesImpl(
    const Descriptor &query,
    const Descriptor &candidate,
    int max_matches,
    bool allow_jointly_absent_upper) const
{
    std::vector<YawMatch> matches;
    if (!validateDescriptorShape(query) || !validateDescriptorShape(candidate))
        return matches;

    const int requested_matches =
        std::max(1, max_matches > 0 ? max_matches : config_.yaw_top_k);

    // Match the released Scan Context search: first align the compact sector
    // key over all circular shifts, then evaluate the full descriptor only in
    // the local 10% window around that coarse shift. For 60 sectors this is
    // the coarse shift plus three neighbors on either side (7 exact scores).
    const Eigen::RowVectorXd query_sector_key = makeSectorKey(query);
    const Eigen::RowVectorXd candidate_sector_key = makeSectorKey(candidate);
    int coarse_shift = 0;
    double best_sector_key_sq = std::numeric_limits<double>::infinity();
    for (int shift = 0; shift < config_.num_sectors; ++shift)
    {
        double diff_sq = 0.0;
        for (int col = 0; col < config_.num_sectors; ++col)
        {
            const int source_col =
                (col - shift + config_.num_sectors) % config_.num_sectors;
            const double diff =
                query_sector_key(col) - candidate_sector_key(source_col);
            diff_sq += diff * diff;
        }
        if (diff_sq < best_sector_key_sq)
        {
            best_sector_key_sq = diff_sq;
            coarse_shift = shift;
        }
    }

    const int search_radius = static_cast<int>(
        std::round(0.5 * 0.1 * static_cast<double>(config_.num_sectors)));
    std::vector<YawMatch> shift_distances;
    shift_distances.reserve(1 + 2 * search_radius);
    for (int offset = -search_radius; offset <= search_radius; ++offset)
    {
        const int shift =
            (coarse_shift + offset + config_.num_sectors) % config_.num_sectors;
        Descriptor shifted_candidate = circularShiftColumns(candidate, shift);
        const double current_distance = directDistance(
            query, shifted_candidate, allow_jointly_absent_upper);
        if (!std::isfinite(current_distance))
            continue;

        YawMatch match;
        match.distance = current_distance;
        match.sector_shift = shift;
        match.yaw_shift_rad =
            static_cast<double>(shift) * kTwoPi / static_cast<double>(config_.num_sectors);
        shift_distances.push_back(match);
    }

    if (shift_distances.empty())
        return matches;

    std::sort(shift_distances.begin(), shift_distances.end(),
              [](const YawMatch &lhs, const YawMatch &rhs) {
                  return lhs.distance < rhs.distance;
              });

    auto circular_sector_distance = [this](int lhs, int rhs) {
        int diff = std::abs(lhs - rhs) % config_.num_sectors;
        return std::min(diff, config_.num_sectors - diff);
    };
    auto append_with_spacing = [&](const YawMatch &match, int min_sector_gap) {
        for (const auto &selected : matches)
        {
            if (selected.sector_shift == match.sector_shift)
                return false;
            if (min_sector_gap > 0 &&
                circular_sector_distance(selected.sector_shift, match.sector_shift) <= min_sector_gap)
                return false;
        }
        matches.push_back(match);
        return true;
    };

    for (const auto &match : shift_distances)
    {
        if (static_cast<int>(matches.size()) >= requested_matches)
            break;
        append_with_spacing(match, 1);
    }

    for (const auto &match : shift_distances)
    {
        if (static_cast<int>(matches.size()) >= requested_matches)
            break;
        append_with_spacing(match, 0);
    }

    std::sort(matches.begin(), matches.end(),
              [](const YawMatch &lhs, const YawMatch &rhs) {
                  return lhs.distance < rhs.distance;
              });
    return matches;
}

Eigen::VectorXd Database::makeRingKey(const Descriptor &descriptor) const
{
    Eigen::VectorXd key = Eigen::VectorXd::Zero(descriptor.values.rows());
    for (int row = 0; row < descriptor.values.rows(); ++row)
    {
        double sum = 0.0;
        int valid_count = 0;
        for (int col = 0; col < descriptor.values.cols(); ++col)
        {
            if (!descriptor.valid(row, col))
                continue;
            sum += descriptor.values(row, col) +
                   config_.retrieval_height_offset;
            ++valid_count;
        }
        if (valid_count > 0)
            key(row) = sum / static_cast<double>(descriptor.values.cols());
    }
    return key;
}

Eigen::RowVectorXd Database::makeSectorKey(const Descriptor &descriptor) const
{
    Eigen::RowVectorXd key = Eigen::RowVectorXd::Zero(descriptor.values.cols());
    for (int col = 0; col < descriptor.values.cols(); ++col)
    {
        double sum = 0.0;
        int valid_count = 0;
        for (int row = 0; row < descriptor.values.rows(); ++row)
        {
            if (!descriptor.valid(row, col))
                continue;
            sum += descriptor.values(row, col) +
                   config_.retrieval_height_offset;
            ++valid_count;
        }
        if (valid_count > 0)
            key(col) = sum / static_cast<double>(descriptor.values.rows());
    }
    return key;
}

double Database::directDistance(const Descriptor &query,
                                const Descriptor &candidate_shifted,
                                bool allow_jointly_absent_upper) const
{
    struct LayerScore
    {
        bool query_supported = false;
        bool candidate_supported = false;
        bool comparable = false;
        bool jointly_absent_omittable = true;
        double distance = 1.0;
    };

    auto layer_distance = [&](int row_offset, int row_count,
                              bool jointly_absent_omittable) {
        LayerScore score;
        score.jointly_absent_omittable =
            jointly_absent_omittable;
        int effective_columns = 0;
        int query_supported_columns = 0;
        int candidate_supported_columns = 0;
        bool query_has_observation = false;
        bool candidate_has_observation = false;
        double similarity_sum = 0.0;

        for (int col = 0; col < query.values.cols(); ++col)
        {
            int joint_count = 0;
            int query_valid_count = 0;
            int candidate_valid_count = 0;
            double dot = 0.0;
            double query_sq_norm = 0.0;
            double candidate_sq_norm = 0.0;
            for (int row = row_offset; row < row_offset + row_count; ++row)
            {
                query_valid_count += query.valid(row, col) ? 1 : 0;
                candidate_valid_count += candidate_shifted.valid(row, col) ? 1 : 0;
                if (!query.valid(row, col) || !candidate_shifted.valid(row, col))
                    continue;
                const double query_value =
                    query.values(row, col) + config_.retrieval_height_offset;
                const double candidate_value =
                    candidate_shifted.values(row, col) +
                    config_.retrieval_height_offset;
                dot += query_value * candidate_value;
                query_sq_norm += query_value * query_value;
                candidate_sq_norm += candidate_value * candidate_value;
                ++joint_count;
            }
            query_has_observation =
                query_has_observation || query_valid_count > 0;
            candidate_has_observation =
                candidate_has_observation || candidate_valid_count > 0;
            if (query_valid_count >= config_.min_joint_rings)
                ++query_supported_columns;
            if (candidate_valid_count >= config_.min_joint_rings)
                ++candidate_supported_columns;
            if (joint_count < config_.min_joint_rings ||
                query_sq_norm <= 1e-24 || candidate_sq_norm <= 1e-24)
                continue;

            const double value_similarity = dot / std::sqrt(query_sq_norm * candidate_sq_norm);
            const double mask_similarity = static_cast<double>(joint_count) /
                std::sqrt(static_cast<double>(query_valid_count * candidate_valid_count));
            similarity_sum += value_similarity * mask_similarity;
            ++effective_columns;
        }

        // Structural absence means no observed cell at all. A sparse layer
        // that fails min_joint_rings still contains conflicting evidence and
        // must be penalized rather than silently omitted.
        score.query_supported = query_has_observation;
        score.candidate_supported = candidate_has_observation;
        if (effective_columns == 0 ||
            !score.query_supported || !score.candidate_supported)
        {
            return score;
        }
        // The per-sector mask cosine above measures ring overlap only after a
        // sector has enough joint support. Without this second factor, sectors
        // with no overlap disappear from the denominator and a wrong place can
        // score perfectly from one accidental matching sector. This is the
        // cosine similarity of the binary supported-sector masks.
        const double sector_support_similarity =
            static_cast<double>(effective_columns) /
            std::sqrt(static_cast<double>(
                query_supported_columns * candidate_supported_columns));
        score.distance = 1.0 -
            (similarity_sum / static_cast<double>(effective_columns)) *
                std::pow(
                    sector_support_similarity,
                    config_.sector_support_exponent);
        score.comparable = std::isfinite(score.distance);
        return score;
    };

    if (!config_.dual_z_layer_enable)
    {
        const LayerScore score =
            layer_distance(0, query.values.rows(), true);
        if (!score.comparable)
            return std::numeric_limits<double>::infinity();
        return score.distance;
    }

    const LayerScore low =
        layer_distance(0, config_.num_rings, true);
    const LayerScore high =
        layer_distance(
            config_.num_rings, config_.num_rings,
            allow_jointly_absent_upper);
    const double low_weight = std::max(0.0, config_.dual_z_low_weight);
    const double high_weight = std::max(0.0, config_.dual_z_high_weight);

    double weight_sum = 0.0;
    double weighted_distance = 0.0;
    bool hard_layer_mismatch = false;
    auto accumulate_layer = [&](const LayerScore &score, double weight) {
        if (weight <= 0.0)
            return;
        // A channel that is structurally absent on both sides carries no
        // evidence and is omitted. If only one side has support, or both sides
        // have support without enough joint cells at this yaw, retain the
        // channel with the maximum normalized distance instead of allowing the
        // missing observation to improve the match.
        if (!score.query_supported && !score.candidate_supported &&
            score.jointly_absent_omittable)
            return;
        if (!score.comparable)
        {
            hard_layer_mismatch = true;
            return;
        }
        weighted_distance += weight * score.distance;
        weight_sum += weight;
    };
    accumulate_layer(low, low_weight);
    accumulate_layer(high, high_weight);
    if (hard_layer_mismatch)
        return 1.0;
    if (weight_sum <= 1e-12)
        return std::numeric_limits<double>::infinity();
    return weighted_distance / weight_sum;
}

Eigen::Array<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic>
Database::makeBoundaryMask(const Descriptor &descriptor) const
{
    if (!config_.dual_z_layer_enable ||
        config_.vertical_boundary_margin <= 0.0 ||
        descriptor.values.rows() != descriptorRows(config_) ||
        descriptor.values.cols() != config_.num_sectors ||
        descriptor.valid.rows() != descriptorRows(config_) ||
        descriptor.valid.cols() != config_.num_sectors)
    {
        return {};
    }

    Eigen::Array<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic> boundary =
        Eigen::Array<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic>::Zero(
            config_.num_rings, config_.num_sectors);

    constexpr double kBoundaryEpsilon = 1e-9;
    for (int ring = 0; ring < config_.num_rings; ++ring)
    {
        const int upper_ring = config_.num_rings + ring;
        for (int sector = 0; sector < config_.num_sectors; ++sector)
        {
            if (!descriptor.valid(ring, sector) ||
                !descriptor.valid(upper_ring, sector))
            {
                continue;
            }
            const double split_height = effectiveDualZSplitHeight(config_);
            const double lower_gap =
                split_height - descriptor.values(ring, sector);
            const double upper_gap =
                descriptor.values(upper_ring, sector) - split_height;
            if (lower_gap >= -kBoundaryEpsilon &&
                upper_gap >= -kBoundaryEpsilon &&
                lower_gap <= config_.vertical_boundary_margin &&
                upper_gap <= config_.vertical_boundary_margin)
            {
                boundary(ring, sector) = 1U;
            }
        }
    }
    return boundary;
}

Descriptor Database::circularShiftColumns(const Descriptor &descriptor, int shift) const
{
    if (descriptor.values.cols() == 0)
        return descriptor;

    shift %= descriptor.values.cols();
    if (shift < 0)
        shift += descriptor.values.cols();
    if (shift == 0)
        return descriptor;

    Descriptor shifted;
    shifted.values = Eigen::MatrixXd::Zero(descriptor.values.rows(), descriptor.values.cols());
    shifted.valid = Eigen::Array<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic>::Zero(
        descriptor.valid.rows(), descriptor.valid.cols());
    if (descriptor.boundary.rows() == config_.num_rings &&
        descriptor.boundary.cols() == descriptor.values.cols())
    {
        shifted.boundary =
            Eigen::Array<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic>::Zero(
                descriptor.boundary.rows(), descriptor.boundary.cols());
    }
    for (int col = 0; col < descriptor.values.cols(); ++col)
    {
        const int new_col = (col + shift) % descriptor.values.cols();
        shifted.values.col(new_col) = descriptor.values.col(col);
        shifted.valid.col(new_col) = descriptor.valid.col(col);
        if (shifted.boundary.size() > 0)
            shifted.boundary.col(new_col) = descriptor.boundary.col(col);
    }
    return shifted;
}

double Database::estimateVerticalShift(const Descriptor &query,
                                       const Descriptor &candidate,
                                       int sector_shift) const
{
    if (!config_.dual_z_layer_enable || !config_.vertical_estimation_enable)
        return 0.0;

    const Descriptor shifted_candidate = circularShiftColumns(candidate, sector_shift);
    const bool have_query_boundary =
        query.boundary.rows() == config_.num_rings &&
        query.boundary.cols() == config_.num_sectors;
    const bool have_candidate_boundary =
        shifted_candidate.boundary.rows() == config_.num_rings &&
        shifted_candidate.boundary.cols() == config_.num_sectors;
    struct CellResidual
    {
        double residual;
        double stability_value;
    };
    std::vector<CellResidual> lower;
    std::vector<CellResidual> upper;
    lower.reserve(static_cast<std::size_t>(config_.num_rings * query.values.cols()));
    upper.reserve(static_cast<std::size_t>(config_.num_rings * query.values.cols()));

    for (int row = 0; row < query.values.rows(); ++row)
    {
        auto &channel = row < config_.num_rings ? lower : upper;
        for (int col = 0; col < query.values.cols(); ++col)
        {
            if (!query.valid(row, col) || !shifted_candidate.valid(row, col))
                continue;
            const int base_ring = row % config_.num_rings;
            if ((have_query_boundary &&
                 query.boundary(base_ring, col) != 0U) ||
                (have_candidate_boundary &&
                 shifted_candidate.boundary(base_ring, col) != 0U))
            {
                // Retrieval already used the original descriptor values.
                // Only Delta-z excludes ambiguous split-boundary heights.
                continue;
            }
            const double query_value = query.values(row, col);
            const double candidate_value = shifted_candidate.values(row, col);
            const double residual = candidate_value - query_value;
            if (!std::isfinite(residual))
                continue;

            // Lower cells store maxima, so a pair is stable only when both
            // maxima are low. Upper cells store minima, so both minima must be
            // high. max/min below implement those conservative pair scores.
            const double stability_value = row < config_.num_rings
                ? std::max(query_value, candidate_value)
                : std::min(query_value, candidate_value);
            channel.push_back({residual, stability_value});
        }
    }

    std::vector<std::pair<double, double>> residuals;
    residuals.reserve(lower.size() + upper.size());
    auto retain_stable_half = [this, &residuals](
                                  std::vector<CellResidual> &channel,
                                  double weight,
                                  bool ascending) {
        if (channel.empty() || weight <= 0.0)
            return;
        std::sort(channel.begin(), channel.end(),
                  [ascending](const CellResidual &lhs, const CellResidual &rhs) {
                      return ascending
                          ? lhs.stability_value < rhs.stability_value
                          : lhs.stability_value > rhs.stability_value;
                  });
        const std::size_t keep = std::max<std::size_t>(
            1, static_cast<std::size_t>(std::ceil(
                   config_.vertical_stable_fraction * static_cast<double>(channel.size()))));
        for (std::size_t i = 0; i < keep; ++i)
            residuals.emplace_back(channel[i].residual, weight);
    };
    retain_stable_half(lower, config_.dual_z_low_weight, true);
    retain_stable_half(upper, config_.dual_z_high_weight, false);
    if (residuals.empty())
        return 0.0;

    std::sort(residuals.begin(), residuals.end(),
              [](const auto &lhs, const auto &rhs) {
                  return lhs.first < rhs.first;
              });
    double total_weight = 0.0;
    for (const auto &residual : residuals)
        total_weight += residual.second;
    const double half_weight = 0.5 * total_weight;
    double accumulated_weight = 0.0;
    for (const auto &residual : residuals)
    {
        accumulated_weight += residual.second;
        if (accumulated_weight >= half_weight)
            return residual.first;
    }
    return residuals.back().first;
}

bool Database::descriptorHasUpperObservation(
    const Descriptor &descriptor) const
{
    if (!config_.dual_z_layer_enable ||
        !validateDescriptorShape(descriptor))
    {
        return false;
    }
    return (descriptor.valid.bottomRows(config_.num_rings) != 0).any();
}

void Database::ensureObservationStats() const
{
    if (!observation_stats_dirty_)
        return;
    entries_with_upper_observation_ = 0;
    entry_allows_absent_upper_fallback_.assign(entries_.size(), 0);
    std::vector<std::uint8_t> has_upper(entries_.size(), 0);
    for (std::size_t i = 0; i < entries_.size(); ++i)
    {
        if (descriptorHasUpperObservation(entries_[i].descriptor))
        {
            has_upper[i] = 1;
            ++entries_with_upper_observation_;
        }
    }
    if (entries_.empty())
    {
        observation_stats_dirty_ = false;
        return;
    }

    const double global_upper_fraction =
        static_cast<double>(entries_with_upper_observation_) /
        static_cast<double>(entries_.size());
    if (global_upper_fraction <=
        config_.absent_upper_fallback_max_local_fraction)
    {
        std::fill(
            entry_allows_absent_upper_fallback_.begin(),
            entry_allows_absent_upper_fallback_.end(), 1);
        observation_stats_dirty_ = false;
        return;
    }

    const double radius_sq =
        config_.absent_upper_fallback_radius *
        config_.absent_upper_fallback_radius;
    for (std::size_t i = 0; i < entries_.size(); ++i)
    {
        int local_count = 0;
        int local_upper_count = 0;
        for (std::size_t j = 0; j < entries_.size(); ++j)
        {
            const double dx = entries_[i].pose.x - entries_[j].pose.x;
            const double dy = entries_[i].pose.y - entries_[j].pose.y;
            if (dx * dx + dy * dy > radius_sq)
                continue;
            ++local_count;
            local_upper_count += has_upper[j] != 0 ? 1 : 0;
        }
        if (local_count < config_.absent_upper_fallback_min_keyframes)
            continue;
        const double local_upper_fraction =
            static_cast<double>(local_upper_count) /
            static_cast<double>(local_count);
        entry_allows_absent_upper_fallback_[i] =
            local_upper_fraction <=
                    config_.absent_upper_fallback_max_local_fraction
                ? 1
                : 0;
    }
    observation_stats_dirty_ = false;
}

void Database::refreshKeys(Entry &entry) const
{
    if (!validateDescriptorShape(entry.descriptor))
    {
        entry.ring_key.resize(0);
        entry.sector_key.resize(0);
        return;
    }
    entry.descriptor.boundary = makeBoundaryMask(entry.descriptor);
    entry.ring_key = makeRingKey(entry.descriptor);
    entry.sector_key = makeSectorKey(entry.descriptor);
    entry.pose.yaw = normalizeAngle(entry.pose.yaw);
    entry.pose.canonical_yaw = normalizeAngle(entry.pose.canonical_yaw);
}

bool Database::validateDescriptorShape(const Descriptor &descriptor) const
{
    return descriptor.values.rows() == descriptorRows(config_) &&
           descriptor.values.cols() == config_.num_sectors &&
           descriptor.valid.rows() == descriptor.values.rows() &&
           descriptor.valid.cols() == descriptor.values.cols() &&
           descriptor.values.allFinite() &&
           (descriptor.valid <= 1).all();
}

double makeCandidateSeedYaw(double base_yaw, double yaw_shift_rad)
{
    return normalizeAngle(base_yaw - yaw_shift_rad);
}

bool makeGravityCanonicalRotation(const Eigen::Vector3d &up,
                                  Eigen::Matrix3d &rotation)
{
    rotation.setIdentity();
    if (!up.allFinite() || up.squaredNorm() < 1e-12)
        return false;

    const Eigen::Vector3d up_unit = up.normalized();
    const double alignment = std::clamp(up_unit.z(), -1.0, 1.0);
    if (alignment > 1.0 - 1e-12)
        return true;
    if (alignment < -1.0 + 1e-12)
    {
        // A 180-degree minimum rotation is not unique. Match the offline and
        // manual-loop implementations with a stable +X-axis convention.
        rotation = Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitX()).toRotationMatrix();
        return true;
    }
    Eigen::Quaterniond q = Eigen::Quaterniond::FromTwoVectors(
        up_unit, Eigen::Vector3d::UnitZ());
    if (!q.coeffs().allFinite() || q.squaredNorm() < 1e-12)
        return false;
    q.normalize();
    rotation = q.toRotationMatrix();
    return rotation.allFinite() &&
           std::abs(rotation.determinant() - 1.0) < 1e-9;
}

PointCloud gravityCanonicalize(const PointCloud &scan,
                               const Eigen::Matrix3d &rotation,
                               const Eigen::Vector3d &origin)
{
    PointCloud canonical;
    if (!rotation.allFinite() || !origin.allFinite())
        return canonical;

    canonical.reserve(scan.size());
    for (const auto &point : scan.points)
    {
        if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z))
            continue;
        const Eigen::Vector3d transformed =
            rotation * (Eigen::Vector3d(point.x, point.y, point.z) - origin);
        if (!transformed.allFinite())
            continue;
        PointType out = point;
        out.x = static_cast<float>(transformed.x());
        out.y = static_cast<float>(transformed.y());
        out.z = static_cast<float>(transformed.z());
        canonical.push_back(out);
    }
    canonical.width = static_cast<std::uint32_t>(canonical.size());
    canonical.height = 1;
    canonical.is_dense = true;
    return canonical;
}

}  // namespace scan_context
}  // namespace fast_lio
