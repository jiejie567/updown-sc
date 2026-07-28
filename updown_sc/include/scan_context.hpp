#pragma once

#include <Eigen/Dense>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <cstdint>
#include <string>
#include <vector>

namespace fast_lio
{
namespace scan_context
{

using PointType = pcl::PointXYZINormal;
using PointCloud = pcl::PointCloud<PointType>;

struct Config
{
    int num_rings = 20;
    int num_sectors = 60;
    double max_radius = 80.0;
    bool dual_z_layer_enable = false;
    // Physical split height above the ground plane. Descriptor values are
    // stored in the same ground-relative height frame.
    double dual_z_split_height = 2.5;
    // Estimate one map-level physical split from all mapping keyframes, then
    // freeze it for both stored descriptors and future queries. The selected
    // value is persisted as dual_z_split_height in the SCD header.
    bool dual_z_split_auto = false;
    double dual_z_split_auto_min = 1.5;
    double dual_z_split_auto_max = 4.5;
    double dual_z_split_auto_bin_size = 0.1;
    double dual_z_split_auto_histogram_max = 8.0;
    double dual_z_split_auto_min_layer_fraction = 0.05;
    int dual_z_split_auto_min_keyframes = 20;
    // Positive vertical distance from the descriptor-frame origin to the
    // ground. Descriptor construction converts point-frame z to ground height
    // with z_ground = z + origin_height_from_ground.
    double origin_height_from_ground = 0.0;
    double dual_z_low_weight = 0.3;
    double dual_z_high_weight = 0.7;
    int min_joint_rings = 2;
    // Jointly absent upper envelopes are omitted only in a candidate's local
    // map neighborhood when upper observations are sufficiently rare.
    double absent_upper_fallback_max_local_fraction = 0.05;
    double absent_upper_fallback_radius = 10.0;
    int absent_upper_fallback_min_keyframes = 3;
    // Runtime-only numerical offset applied to valid descriptor heights when
    // constructing retrieval keys and cosine distances. Stored descriptors
    // and Delta-z estimation retain ground-relative physical heights.
    double retrieval_height_offset = 0.1;
    // The ring-mask overlap already penalizes local missing observations.
    // A square-root sector-support penalty keeps protection against accidental
    // few-sector matches without penalizing sparse scans twice as strongly.
    double sector_support_exponent = 0.5;
    // Cells whose lower maximum and upper minimum both lie close to the split
    // have unreliable layer identity and are excluded only from Delta-z
    // estimation. Retrieval keeps the original dual-envelope score.
    double vertical_boundary_margin = 0.1;
    // Metadata describing the point frame used to build stored descriptors.
    // Canonicalization itself is performed by the caller because gravity comes
    // from the LiDAR-inertial estimator rather than Scan Context.
    bool gravity_canonicalized = false;
    // Estimate z only after the legacy Scan Context candidate and yaw have
    // been fixed, so height cannot change retrieval ranking.
    bool vertical_estimation_enable = false;
    double vertical_correction_min = -1.5;
    double vertical_correction_max = 1.5;
    // Fraction of jointly valid cells retained independently in each envelope:
    // lowest lower-envelope maxima and highest upper-envelope minima.
    double vertical_stable_fraction = 0.5;
    int candidate_top_k = 5;
    int yaw_top_k = 3;
    double distance_thresh = 0.5;
};

// Kept for source compatibility with callers that used the pre-V7 name.
// Since V7 descriptors are ground-normalized, the effective descriptor-frame
// split is the physical split itself.
double effectiveDualZSplitHeight(const Config &config);

struct AdaptiveSplitResult
{
    bool adapted = false;
    bool dual_layer_enabled = true;
    double split_height = 2.5;
    std::size_t keyframe_count = 0;
    std::uint64_t support_count = 0;
    double lower_fraction = 0.0;
    double upper_fraction = 0.0;
    double between_class_variance = 0.0;
};

// Accumulates a cell-balanced height histogram: within one keyframe each
// ring-sector cell votes at most once for a height bin. This prevents dense
// surfaces or LiDAR sampling patterns from dominating the map-level split.
class AdaptiveSplitEstimator
{
public:
    explicit AdaptiveSplitEstimator(const Config &config = Config());

    void setConfig(const Config &config);
    void clear();
    void addScan(const PointCloud &scan);
    AdaptiveSplitResult estimate() const;

private:
    Config config_;
    std::vector<std::uint64_t> histogram_;
    std::size_t keyframe_count_ = 0;
};

struct Descriptor
{
    Eigen::MatrixXd values;
    Eigen::Array<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic> valid;
    // Runtime cache over base ring-sector cells. It is derived from values and
    // valid masks and deliberately not serialized in the SCD database.
    Eigen::Array<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic> boundary;
};

struct Pose
{
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    double roll = 0.0;
    double pitch = 0.0;
    double yaw = 0.0;
    // Heading of the gravity-canonical descriptor frame in the map frame.
    // This can differ slightly from ZYX yaw when roll and pitch coexist.
    double canonical_yaw = 0.0;
};

struct Entry
{
    double stamp = 0.0;
    Pose pose;
    Descriptor descriptor;
    Eigen::VectorXd ring_key;
    Eigen::RowVectorXd sector_key;
};

struct YawMatch
{
    double distance = 0.0;
    int sector_shift = 0;
    double yaw_shift_rad = 0.0;
    double coarse_vertical_shift = 0.0;
    double vertical_shift = 0.0;
};

struct Candidate
{
    int index = -1;
    double distance = 0.0;
    int sector_shift = 0;
    double yaw_shift_rad = 0.0;
    double coarse_vertical_shift = 0.0;
    double vertical_shift = 0.0;
    Pose pose;
    std::vector<YawMatch> yaw_matches;
};

class Database
{
public:
    explicit Database(const Config &config = Config());

    const Config &config() const;
    void setConfig(const Config &config);
    void clear();
    bool empty() const;
    std::size_t size() const;
    const std::vector<Entry> &entries() const;
    bool legacyMasksInferred() const;
    std::size_t absentUpperFallbackEntryCount() const;

    Descriptor makeDescriptor(const PointCloud &scan,
                              double vertical_shift = 0.0) const;
    void addEntry(double stamp, const Pose &pose, const Descriptor &descriptor);
    std::vector<Candidate> query(const Descriptor &descriptor,
                                 bool apply_distance_threshold = true) const;
    std::vector<Candidate> queryWithVerticalEstimation(
        const Descriptor &descriptor,
        bool apply_distance_threshold = true) const;
    std::vector<Candidate> query(const PointCloud &scan,
                                 bool apply_distance_threshold = true) const;

    bool save(const std::string &path, std::string *error = nullptr) const;
    bool load(const std::string &path, std::string *error = nullptr);

    double distance(const Descriptor &query,
                    const Descriptor &candidate,
                    int *sector_shift = nullptr) const;
    std::vector<YawMatch> yawMatches(const Descriptor &query,
                                     const Descriptor &candidate,
                                     int max_matches = -1) const;

private:
    Eigen::VectorXd makeRingKey(const Descriptor &descriptor) const;
    Eigen::RowVectorXd makeSectorKey(const Descriptor &descriptor) const;
    double directDistance(const Descriptor &query,
                          const Descriptor &candidate_shifted,
                          bool allow_jointly_absent_upper) const;
    std::vector<YawMatch> yawMatchesImpl(
        const Descriptor &query,
        const Descriptor &candidate,
        int max_matches,
        bool allow_jointly_absent_upper) const;
    Descriptor circularShiftColumns(const Descriptor &descriptor, int shift) const;
    double estimateVerticalShift(const Descriptor &query,
                                 const Descriptor &candidate,
                                 int sector_shift) const;
    Eigen::Array<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic>
    makeBoundaryMask(const Descriptor &descriptor) const;
    bool descriptorHasUpperObservation(const Descriptor &descriptor) const;
    void ensureObservationStats() const;
    void refreshKeys(Entry &entry) const;
    bool validateDescriptorShape(const Descriptor &descriptor) const;

    Config config_;
    std::vector<Entry> entries_;
    bool legacy_masks_inferred_ = false;
    mutable std::size_t entries_with_upper_observation_ = 0;
    mutable std::vector<std::uint8_t>
        entry_allows_absent_upper_fallback_;
    mutable bool observation_stats_dirty_ = true;
};

double makeCandidateSeedYaw(double base_yaw, double yaw_shift_rad);

// Returns the minimum rotation that maps a measured up direction onto +Z.
// No horizontal heading is selected, so residual yaw remains a circular
// Scan Context sector shift.
bool makeGravityCanonicalRotation(const Eigen::Vector3d &up,
                                  Eigen::Matrix3d &rotation);

PointCloud gravityCanonicalize(const PointCloud &scan,
                               const Eigen::Matrix3d &rotation,
                               const Eigen::Vector3d &origin = Eigen::Vector3d::Zero());

}  // namespace scan_context
}  // namespace fast_lio
