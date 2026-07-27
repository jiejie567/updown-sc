#include "scan_context.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>

namespace sc = fast_lio::scan_context;

namespace
{

void require(bool condition, const std::string &message)
{
    if (!condition)
        throw std::runtime_error(message);
}

sc::Config testConfig()
{
    sc::Config config;
    config.num_rings = 3;
    config.num_sectors = 4;
    config.max_radius = 9.0;
    config.dual_z_layer_enable = true;
    config.dual_z_split_height = 1.0;
    config.retrieval_height_offset = 0.0;
    config.min_joint_rings = 2;
    config.candidate_top_k = 1;
    config.yaw_top_k = 1;
    return config;
}

sc::PointType point(float x, float y, float z);

void testPlatformOriginHeightNormalizesDescriptor()
{
    sc::Config config = testConfig();
    config.dual_z_split_height = 2.5;
    config.origin_height_from_ground = 1.5;
    require(std::abs(sc::effectiveDualZSplitHeight(config) - 2.5) < 1e-12,
            "ground-normalized descriptor did not retain the physical split");

    sc::Database database(config);
    sc::PointCloud scan;
    scan.push_back(point(1.0F, 0.0F, 0.8F));
    scan.push_back(point(4.0F, 0.0F, 1.2F));
    const sc::Descriptor descriptor = database.makeDescriptor(scan);
    require(descriptor.valid(0, 0) == 1,
            "point below the ground-relative split left the lower envelope");
    require(descriptor.valid(4, 0) == 1,
            "point above the ground-relative split did not enter the upper envelope");
    require(std::abs(descriptor.values(0, 0) - 2.3) < 1e-6,
            "lower-envelope value was not converted to ground-relative height");
    require(std::abs(descriptor.values(4, 0) - 2.7) < 1e-6,
            "upper-envelope value was not converted to ground-relative height");
}

void testDefaultRetrievalHeightOffset()
{
    require(std::abs(sc::Config().retrieval_height_offset - 0.1) < 1e-12,
            "retrieval height offset default is not 0.1 m");
}

void testIndependentPlatformHeightsProduceCommonGroundDescriptor()
{
    sc::Config map_config = testConfig();
    map_config.dual_z_split_height = 2.5;
    map_config.origin_height_from_ground = 1.5;
    map_config.retrieval_height_offset = 0.1;
    map_config.distance_thresh = 1.0;
    map_config.vertical_estimation_enable = true;
    sc::Config query_config = map_config;
    query_config.origin_height_from_ground = 1.1;

    sc::PointCloud map_scan;
    sc::PointCloud query_scan;
    for (const float radius : {1.0F, 4.0F, 7.0F})
    {
        map_scan.push_back(point(radius, 0.0F, -0.5F));
        map_scan.push_back(point(radius, 0.0F, 1.5F));
        query_scan.push_back(point(radius, 0.0F, -0.1F));
        query_scan.push_back(point(radius, 0.0F, 1.9F));
    }

    sc::Database map_database(map_config);
    map_database.addEntry(
        1.0, sc::Pose{}, map_database.makeDescriptor(map_scan));
    sc::Database query_builder(query_config);
    const sc::Descriptor query_descriptor =
        query_builder.makeDescriptor(query_scan);
    const auto candidates = map_database.queryWithVerticalEstimation(
        query_descriptor, false);
    require(candidates.size() == 1,
            "ground-normalized platform descriptors produced no retrieval candidate");
    require(std::abs(candidates.front().distance) < 1e-12,
            "equal physical heights did not produce equal descriptors");
    require(std::abs(candidates.front().vertical_shift) < 1e-5,
            "ground normalization left a platform-height residual");
}

sc::PointType point(float x, float y, float z)
{
    sc::PointType result;
    result.x = x;
    result.y = y;
    result.z = z;
    return result;
}

void testEnvelopeMasks()
{
    sc::Database database(testConfig());
    sc::PointCloud scan;
    scan.push_back(point(1.0F, 0.0F, -2.0F));
    scan.push_back(point(4.0F, 0.0F, 3.0F));
    scan.push_back(point(4.0F, 0.0F, 4.0F));

    const sc::Descriptor descriptor = database.makeDescriptor(scan);
    require(descriptor.valid(0, 0) == 1, "negative lower envelope must remain valid");
    require(std::abs(descriptor.values(0, 0) + 2.0) < 1e-12,
            "signed negative lower-envelope height changed");
    require(descriptor.valid(4, 0) == 1, "upper envelope mask was not set");
    require(std::abs(descriptor.values(4, 0) - 3.0) < 1e-12,
            "upper envelope must keep the signed minimum height");
    require(descriptor.valid(2, 3) == 0, "empty cell was marked valid");
    require(std::abs(descriptor.values(2, 3)) < 1e-12, "empty cell value must remain a harmless placeholder");
}

void testVerticalShiftChangesEnvelopeMasks()
{
    sc::Config config = testConfig();
    sc::Database database(config);
    sc::PointCloud scan;
    scan.push_back(point(1.0F, 0.0F, 0.8F));

    const sc::Descriptor unshifted = database.makeDescriptor(scan, 0.0);
    const sc::Descriptor shifted = database.makeDescriptor(scan, 0.3);
    require(unshifted.valid(0, 0) == 1 && unshifted.valid(3, 0) == 0,
            "unshifted point was assigned to the wrong vertical envelope");
    require(shifted.valid(0, 0) == 0 && shifted.valid(3, 0) == 1,
            "vertical hypothesis did not rebuild the split-dependent masks");
}

void testYawConditionedStableHeightEstimation()
{
    sc::Config config = testConfig();
    config.vertical_estimation_enable = true;
    config.vertical_correction_min = -1.0;
    config.vertical_correction_max = 1.0;
    config.distance_thresh = 1.0;
    sc::Database database(config);

    sc::PointCloud map_scan;
    sc::PointCloud query_scan;
    const float radii[] = {1.0F, 4.0F, 7.0F};
    const float low_query_z[] = {-0.5F, -0.2F, 0.2F};
    const float high_query_z[] = {2.0F, 2.3F, 2.6F};
    constexpr float expected_dz = 0.6F;
    const float map_angle = 10.0F * static_cast<float>(M_PI) / 180.0F;
    const float query_angle = 100.0F * static_cast<float>(M_PI) / 180.0F;
    for (int i = 0; i < 3; ++i)
    {
        // The query is rotated +90 degrees relative to the stored descriptor.
        query_scan.push_back(point(
            radii[i] * std::cos(query_angle), radii[i] * std::sin(query_angle), low_query_z[i]));
        query_scan.push_back(point(
            radii[i] * std::cos(query_angle), radii[i] * std::sin(query_angle), high_query_z[i]));
        map_scan.push_back(point(
            radii[i] * std::cos(map_angle), radii[i] * std::sin(map_angle), low_query_z[i] + expected_dz));
        map_scan.push_back(point(
            radii[i] * std::cos(map_angle), radii[i] * std::sin(map_angle), high_query_z[i] + expected_dz));
    }

    sc::Pose pose;
    pose.z = 3.0;
    database.addEntry(1.0, pose, database.makeDescriptor(map_scan));
    const auto baseline_candidates = database.query(
        database.makeDescriptor(query_scan), false);
    const auto candidates = database.query(query_scan, false);
    require(candidates.size() == 1, "yaw-conditioned height query returned no candidate");
    require(baseline_candidates.size() == candidates.size(),
            "height estimation changed the legacy candidate count");
    require(baseline_candidates.front().index == candidates.front().index &&
            baseline_candidates.front().sector_shift == candidates.front().sector_shift &&
            std::abs(baseline_candidates.front().distance - candidates.front().distance) < 1e-12,
            "height estimation changed the SC-style candidate or yaw ranking");
    require(candidates.front().sector_shift == 1,
            "legacy SCD query selected the wrong circular sector shift");
    require(std::abs(candidates.front().vertical_shift - expected_dz) < 1e-5,
            "one-pass stable vertical estimate has the wrong value or sign");
    require(std::abs((pose.z + candidates.front().vertical_shift) - 3.6) < 1e-5,
            "candidate pose z plus Delta-z did not form the paper ICP seed");
}

void testStableHalfRejectsSplitBoundaryResiduals()
{
    sc::Config config = testConfig();
    config.vertical_estimation_enable = true;
    config.vertical_stable_fraction = 0.5;
    config.vertical_correction_min = -1.0;
    config.vertical_correction_max = 1.0;
    config.distance_thresh = 1.0;
    sc::Database database(config);

    sc::PointCloud map_scan;
    sc::PointCloud query_scan;
    const float radii[] = {1.0F, 4.0F, 7.0F};
    // The third cell in each channel is deliberately close to the split and
    // carries a conflicting residual. The stable 50% must discard it.
    const float lower_query[] = {-1.0F, -0.8F, 0.9F};
    const float lower_map[] = {-0.6F, -0.4F, 0.1F};
    const float upper_query[] = {3.0F, 3.2F, 1.1F};
    const float upper_map[] = {3.4F, 3.6F, 1.9F};
    const float angle = 10.0F * static_cast<float>(M_PI) / 180.0F;
    for (int i = 0; i < 3; ++i)
    {
        const float x = radii[i] * std::cos(angle);
        const float y = radii[i] * std::sin(angle);
        query_scan.push_back(point(x, y, lower_query[i]));
        query_scan.push_back(point(x, y, upper_query[i]));
        map_scan.push_back(point(x, y, lower_map[i]));
        map_scan.push_back(point(x, y, upper_map[i]));
    }

    database.addEntry(1.0, sc::Pose{}, database.makeDescriptor(map_scan));
    const auto candidates = database.query(query_scan, false);
    require(candidates.size() == 1, "stable-half query returned no candidate");
    require(std::abs(candidates.front().vertical_shift - 0.4) < 1e-5,
            "split-boundary cells biased the stable-half Delta-z estimate");
    require(std::abs(candidates.front().coarse_vertical_shift -
                     candidates.front().vertical_shift) < 1e-12,
            "one-pass vertical estimator unexpectedly performed mask refinement");
}

void testJointMaskDistance()
{
    sc::Config config = testConfig();
    config.dual_z_layer_enable = false;
    sc::Database database(config);

    sc::Descriptor query;
    query.values = Eigen::MatrixXd::Zero(3, 4);
    query.valid = Eigen::Array<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic>::Zero(3, 4);
    sc::Descriptor candidate = query;

    query.values(0, 0) = 1.0;
    query.values(1, 0) = 2.0;
    query.values(2, 0) = 100.0;
    query.valid(0, 0) = 1;
    query.valid(1, 0) = 1;
    query.valid(2, 0) = 1;

    candidate.values(0, 0) = 1.0;
    candidate.values(1, 0) = 2.0;
    candidate.values(2, 0) = -500.0;
    candidate.valid(0, 0) = 1;
    candidate.valid(1, 0) = 1;

    int sector_shift = -1;
    const double distance = database.distance(query, candidate, &sector_shift);
    const double expected_mask_penalty = 1.0 - 2.0 / std::sqrt(6.0);
    require(std::abs(distance - expected_mask_penalty) < 1e-12,
            "masked cosine distance lost its validity-overlap penalty");
    require(sector_shift == 0, "unexpected sector shift for identical jointly valid cells");

    candidate.valid(1, 0) = 0;
    require(!std::isfinite(database.distance(query, candidate)),
            "sector with fewer than min_joint_rings should not be compared");
}

void testAbsentDualChannelFallsBackToSupportedEnvelope()
{
    sc::Config config = testConfig();
    config.sector_support_exponent = 0.0;
    sc::Database database(config);

    sc::Descriptor query;
    query.values = Eigen::MatrixXd::Zero(6, 4);
    query.valid =
        Eigen::Array<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic>::Zero(6, 4);
    sc::Descriptor candidate = query;
    for (int ring = 0; ring < 3; ++ring)
    {
        query.values(ring, 0) = 0.5 + ring;
        candidate.values(ring, 0) = 0.5 + ring;
        query.valid(ring, 0) = 1;
        candidate.valid(ring, 0) = 1;
    }

    const double lower_only_distance = database.distance(query, candidate);
    require(std::isfinite(lower_only_distance),
            "dual-envelope match rejected descriptors with no upper structure");
    require(std::abs(lower_only_distance) < 1e-12,
            "absent upper structure changed an identical lower-envelope match");

    // A high channel present only in the candidate is an observation mismatch,
    // not a jointly absent channel. It must retain the configured high-channel
    // penalty instead of being silently discarded.
    for (int ring = 0; ring < 2; ++ring)
    {
        candidate.values(3 + ring, 0) = 3.0 + ring;
        candidate.valid(3 + ring, 0) = 1;
    }
    const double one_sided_distance = database.distance(query, candidate);
    require(std::abs(one_sided_distance - 1.0) < 1e-12,
            "one-sided upper structure escaped the missing-channel penalty");

    candidate.valid.bottomRows(3).setZero();
    query.values(3, 1) = 3.0;
    candidate.values(3, 1) = 3.0;
    query.valid(3, 1) = 1;
    candidate.valid(3, 1) = 1;
    const double sparse_upper_distance = database.distance(query, candidate);
    require(std::abs(sparse_upper_distance - 1.0) < 1e-12,
            "sparse observed upper structure was mistaken for structural absence");

    query.valid.setZero();
    candidate.valid.setZero();
    require(!std::isfinite(database.distance(query, candidate)),
            "descriptors with no jointly usable channel produced a finite score");
}

void testAbsentUpperFallbackRequiresAnUpperFreeMap()
{
    sc::Config config = testConfig();
    config.sector_support_exponent = 0.0;
    config.absent_upper_fallback_max_local_fraction = 0.05;
    config.absent_upper_fallback_radius = 10.0;
    config.absent_upper_fallback_min_keyframes = 1;
    sc::Database database(config);

    sc::Descriptor map_entry;
    map_entry.values = Eigen::MatrixXd::Zero(6, 4);
    map_entry.valid =
        Eigen::Array<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic>::Zero(6, 4);
    for (int ring = 0; ring < 3; ++ring)
    {
        map_entry.values(ring, 0) = 0.5 + ring;
        map_entry.valid(ring, 0) = 1;
    }
    map_entry.values(3, 0) = 3.0;
    map_entry.valid(3, 0) = 1;
    database.addEntry(0.0, sc::Pose{}, map_entry);

    sc::Descriptor query = map_entry;
    sc::Descriptor candidate = map_entry;
    query.valid.bottomRows(3).setZero();
    candidate.valid.bottomRows(3).setZero();
    const double distance = database.distance(query, candidate);
    require(std::abs(distance - 1.0) < 1e-12,
            "mixed-structure map omitted a jointly empty upper channel");
}

void testMixedMapUsesCandidateLocalUpperFallback()
{
    sc::Config config = testConfig();
    config.sector_support_exponent = 0.0;
    config.candidate_top_k = 4;
    config.absent_upper_fallback_max_local_fraction = 0.05;
    config.absent_upper_fallback_radius = 5.0;
    config.absent_upper_fallback_min_keyframes = 2;
    sc::Database database(config);

    sc::Descriptor outdoor;
    outdoor.values = Eigen::MatrixXd::Zero(6, 4);
    outdoor.valid =
        Eigen::Array<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic>::Zero(6, 4);
    for (int ring = 0; ring < 3; ++ring)
    {
        outdoor.values(ring, 0) = 0.5 + ring;
        outdoor.valid(ring, 0) = 1;
    }
    sc::Descriptor indoor = outdoor;
    for (int ring = 0; ring < 2; ++ring)
    {
        indoor.values(3 + ring, 0) = 3.0 + ring;
        indoor.valid(3 + ring, 0) = 1;
    }

    sc::Pose pose;
    pose.x = 0.0;
    database.addEntry(0.0, pose, outdoor);
    pose.x = 1.0;
    database.addEntry(1.0, pose, outdoor);
    pose.x = 100.0;
    database.addEntry(2.0, pose, indoor);
    pose.x = 101.0;
    database.addEntry(3.0, pose, indoor);

    const auto candidates = database.query(outdoor, false);
    require(candidates.size() == 4,
            "mixed-map local fallback removed finite candidates");
    require((candidates[0].index == 0 || candidates[0].index == 1) &&
                std::abs(candidates[0].distance) < 1e-12,
            "outdoor neighborhood did not omit its jointly absent upper layer");
    for (const auto &candidate : candidates)
    {
        if (candidate.index >= 2)
            require(std::abs(candidate.distance - 1.0) < 1e-12,
                    "indoor neighborhood incorrectly used outdoor fallback");
    }
}

void testAdaptivePhysicalSplit()
{
    sc::Config config = testConfig();
    config.dual_z_split_height = 2.5;
    config.dual_z_split_auto = true;
    config.dual_z_split_auto_min = 1.5;
    config.dual_z_split_auto_max = 4.5;
    config.dual_z_split_auto_bin_size = 0.1;
    config.dual_z_split_auto_histogram_max = 6.0;
    config.dual_z_split_auto_min_layer_fraction = 0.05;
    config.dual_z_split_auto_min_keyframes = 4;
    sc::AdaptiveSplitEstimator estimator(config);
    for (int frame = 0; frame < 4; ++frame)
    {
        sc::PointCloud scan;
        for (int sector = 0; sector < 4; ++sector)
        {
            const float angle =
                static_cast<float>(sector * M_PI_2 + 0.1);
            const float radius = 1.0F + static_cast<float>(sector);
            scan.push_back(point(
                radius * std::cos(angle), radius * std::sin(angle),
                0.4F + 0.2F * static_cast<float>(sector)));
            scan.push_back(point(
                radius * std::cos(angle), radius * std::sin(angle),
                3.8F + 0.1F * static_cast<float>(sector)));
        }
        estimator.addScan(scan);
    }
    const sc::AdaptiveSplitResult estimate = estimator.estimate();
    require(estimate.adapted && estimate.dual_layer_enabled,
            "well-supported bimodal map heights did not enable adaptive split");
    require(estimate.split_height >= 1.5 &&
                estimate.split_height <= 4.5,
            "adaptive split escaped its physical bounds");
    require(estimate.lower_fraction > 0.4 &&
                estimate.upper_fraction > 0.4,
            "adaptive split did not separate the two supported height groups");

    sc::AdaptiveSplitEstimator no_upper(config);
    for (int frame = 0; frame < 4; ++frame)
    {
        sc::PointCloud scan;
        scan.push_back(point(1.0F, 0.0F, 0.4F));
        scan.push_back(point(4.0F, 0.0F, 0.8F));
        no_upper.addScan(scan);
    }
    const sc::AdaptiveSplitResult fallback = no_upper.estimate();
    require(fallback.adapted && !fallback.dual_layer_enabled,
            "map without upper structure did not fall back to single-layer SC");
}

void testRetrievalHeightOffsetKeepsSignedDescriptor()
{
    sc::Config config = testConfig();
    config.num_rings = 2;
    config.num_sectors = 1;
    config.dual_z_layer_enable = false;
    config.min_joint_rings = 1;
    config.sector_support_exponent = 0.0;

    sc::Descriptor query;
    query.values = Eigen::MatrixXd::Zero(2, 1);
    query.valid =
        Eigen::Array<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic>::Ones(2, 1);
    query.values(0, 0) = -1.0;
    query.values(1, 0) = 1.0;

    sc::Descriptor candidate = query;
    candidate.values(0, 0) = 0.0;
    candidate.values(1, 0) = 2.0;

    sc::Database signed_database(config);
    const double signed_distance = signed_database.distance(query, candidate);

    config.retrieval_height_offset = 2.0;
    sc::Database offset_database(config);
    const double offset_distance = offset_database.distance(query, candidate);
    require(std::isfinite(signed_distance) && std::isfinite(offset_distance),
            "retrieval height-offset test produced an invalid distance");
    require(offset_distance < signed_distance,
            "positive retrieval offset did not reduce signed-height sensitivity");
    require(std::abs(query.values(0, 0) + 1.0) < 1e-12 &&
            std::abs(candidate.values(1, 0) - 2.0) < 1e-12,
            "retrieval-only offset mutated signed descriptor heights");
}

void testBoundaryFilterDoesNotChangeRetrieval()
{
    sc::Config filtered_config = testConfig();
    filtered_config.vertical_boundary_margin = 0.2;
    sc::Config unfiltered_config = filtered_config;
    unfiltered_config.vertical_boundary_margin = 0.0;
    sc::Database filtered_database(filtered_config);
    sc::Database unfiltered_database(unfiltered_config);

    sc::Descriptor query;
    query.values = Eigen::MatrixXd::Zero(6, 4);
    query.valid = Eigen::Array<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic>::Zero(6, 4);
    sc::Descriptor candidate = query;

    for (int ring = 0; ring < 3; ++ring)
    {
        query.values(ring, 0) = 0.9;
        query.values(3 + ring, 0) = 1.1;
        query.valid(ring, 0) = 1;
        query.valid(3 + ring, 0) = 1;

        candidate.values(ring, 0) = 0.2 + 0.1 * ring;
        candidate.values(3 + ring, 0) = 1.5 + 0.2 * ring;
        candidate.valid(ring, 0) = 1;
        candidate.valid(3 + ring, 0) = 1;
    }

    const double filtered_distance = filtered_database.distance(query, candidate);
    const double unfiltered_distance = unfiltered_database.distance(query, candidate);
    require(std::isfinite(filtered_distance) && std::isfinite(unfiltered_distance),
            "boundary retrieval invariance test produced no comparable sector");
    require(std::abs(filtered_distance - unfiltered_distance) < 1e-12,
            "vertical boundary filtering changed the main retrieval score");
}

void testBoundaryCellsDoNotBiasVerticalShift()
{
    sc::Config config = testConfig();
    config.vertical_boundary_margin = 0.2;
    config.vertical_stable_fraction = 1.0;
    config.distance_thresh = 1.0;
    sc::Database database(config);

    sc::PointCloud map_scan;
    sc::PointCloud query_scan;
    const float radii[] = {1.0F, 4.0F, 7.0F};
    for (float radius : radii)
    {
        query_scan.push_back(point(radius, 0.0F, 0.9F));
        query_scan.push_back(point(radius, 0.0F, 1.1F));
        map_scan.push_back(point(radius, 0.0F, 0.85F));
        map_scan.push_back(point(radius, 0.0F, 1.15F));
    }

    database.addEntry(1.0, sc::Pose{}, database.makeDescriptor(map_scan));
    const auto candidates = database.query(query_scan, false);
    require(candidates.size() == 1, "boundary-only query returned no candidate");
    require(std::abs(candidates.front().vertical_shift) < 1e-12,
            "ambiguous boundary heights leaked into Delta-z estimation");
}

void testSectorSupportRejectsSparseAlias()
{
    sc::Config config = testConfig();
    config.dual_z_layer_enable = false;
    sc::Database database(config);

    sc::Descriptor query;
    query.values = Eigen::MatrixXd::Zero(3, 4);
    query.valid = Eigen::Array<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic>::Zero(3, 4);
    for (int col = 0; col < 4; ++col)
    {
        query.values(0, col) = 1.0;
        query.values(1, col) = 2.0;
        query.valid(0, col) = 1;
        query.valid(1, col) = 1;
    }

    sc::Descriptor dense = query;
    sc::Descriptor sparse = query;
    sparse.values.rightCols(3).setZero();
    sparse.valid.rightCols(3).setZero();

    const double dense_distance = database.distance(query, dense);
    const double sparse_distance = database.distance(query, sparse);
    require(std::abs(dense_distance) < 1e-12,
            "identical dense sector support did not retain zero distance");
    const double expected_sparse_distance = 1.0 - std::sqrt(0.5);
    require(std::abs(sparse_distance - expected_sparse_distance) < 1e-12,
            "sparse accidental sector overlap escaped the global support penalty");
    require(sparse_distance > dense_distance + 0.25,
            "square-root support penalty no longer separates a one-sector alias");
}

void testV7BitsetDatabaseMaskRoundTrip()
{
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() / "fast_lio_scan_context_mask_v7_test.scd";
    std::error_code ec;
    std::filesystem::remove(path, ec);

    sc::Config database_config = testConfig();
    database_config.gravity_canonicalized = true;
    database_config.dual_z_split_height = 2.5;
    database_config.origin_height_from_ground = 1.5;
    sc::Database database(database_config);
    sc::PointCloud scan;
    scan.push_back(point(1.0F, 0.0F, -2.0F));
    sc::Pose stored_pose;
    stored_pose.yaw = 0.25;
    stored_pose.canonical_yaw = 0.4;
    database.addEntry(1.0, stored_pose, database.makeDescriptor(scan));
    std::string error;
    require(database.save(path.string(), &error), "failed to save V7 database: " + error);

    {
        std::ifstream in(path, std::ios::binary);
        std::string magic;
        std::getline(in, magic);
        require(magic == "FAST_LIO_SCAN_CONTEXT_DB_V7", "database was not saved as V7");
        std::string contents(
            (std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        require(contents.find("MASK_BITS 3\n") != std::string::npos,
                "V7 database did not pack its 24-cell mask into three bytes");
    }

    sc::Database loaded(testConfig());
    require(loaded.load(path.string(), &error), "failed to load V7 database: " + error);
    require(!loaded.legacyMasksInferred(), "V7 database was incorrectly marked as legacy");
    require(loaded.size() == 1, "V7 database entry count changed");
    require(loaded.config().min_joint_rings == 2, "V7 database lost min_joint_rings");
    require(std::abs(loaded.config().origin_height_from_ground - 1.5) < 1e-12,
            "V7 database lost the descriptor-origin height");
    require(std::abs(sc::effectiveDualZSplitHeight(loaded.config()) - 2.5) < 1e-12,
            "V7 database changed the physical split");
    require(loaded.config().gravity_canonicalized,
            "V7 database lost its gravity-canonicalization marker");
    require(std::abs(loaded.entries().front().pose.canonical_yaw - 0.4) < 1e-12,
            "V7 database lost descriptor canonical yaw");
    require(loaded.entries().front().descriptor.valid(0, 0) == 1,
            "V7 database lost an explicit valid bit for a negative ground height");
    require(std::abs(loaded.entries().front().descriptor.values(0, 0) + 0.5) < 1e-12,
            "V7 database changed a ground-relative height");

    std::filesystem::remove(path, ec);
}

void testV6SignedHeightMigration()
{
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() /
        "fast_lio_scan_context_v6_ground_height_migration_test.scd";
    {
        std::ofstream out(path, std::ios::binary);
        require(static_cast<bool>(out), "failed to create V6 migration fixture");
        out << "FAST_LIO_SCAN_CONTEXT_DB_V6\n";
        out << "PARAMS 3 4 9 1 2.5 1.5 0.4 0.6 2 1\n";
        out << "ENTRIES 1\n";
        out << "ENTRY 0 1 0 0 0 0 0 0 0\n";
        out << "DESC\n";
        for (int row = 0; row < 6; ++row)
        {
            for (int col = 0; col < 4; ++col)
            {
                if (col > 0)
                    out << ' ';
                out << ((row == 0 && col == 0) ? -2.0 : 0.0);
            }
            out << '\n';
        }
        out << "MASK_BITS 3\n";
        const char packed_mask[3] = {static_cast<char>(0x01), 0, 0};
        out.write(packed_mask, 3);
        out << "\nEND_ENTRY\n";
    }

    sc::Database loaded(testConfig());
    std::string error;
    require(loaded.load(path.string(), &error),
            "failed to load V6 migration fixture: " + error);
    require(std::abs(loaded.entries().front().descriptor.values(0, 0) + 0.5) < 1e-12,
            "V6 signed descriptor height was not migrated to ground height");
    require(std::abs(sc::effectiveDualZSplitHeight(loaded.config()) - 2.5) < 1e-12,
            "V6 physical split changed during migration");

    std::error_code ec;
    std::filesystem::remove(path, ec);
}

void testLegacyV4HeightOffsetMigration()
{
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() / "fast_lio_scan_context_v4_offset_test.scd";
    {
        std::ofstream out(path, std::ios::binary);
        require(static_cast<bool>(out), "failed to create V4 offset fixture");
        out << "FAST_LIO_SCAN_CONTEXT_DB_V4\n";
        out << "PARAMS 3 4 9 2 1 1 0.5 0.5 2 1\n";
        out << "ENTRIES 1\n";
        out << "ENTRY 0 1 0 0 0 0 0 0 0\n";
        out << "DESC\n";
        for (int row = 0; row < 6; ++row)
        {
            for (int col = 0; col < 4; ++col)
            {
                if (col > 0)
                    out << ' ';
                if (row == 0 && col == 0)
                    out << 1.5;  // Signed height -0.5 plus legacy offset 2.0.
                else if (row == 3 && col == 0)
                    out << 3.2;  // Signed height 1.2 plus legacy offset 2.0.
                else
                    out << 0;
            }
            out << '\n';
        }
        out << "MASK_BITS 3\n";
        const char packed_mask[3] = {static_cast<char>(0x01), static_cast<char>(0x10), 0};
        out.write(packed_mask, 3);
        out << "\nEND_ENTRY\n";
    }

    sc::Database loaded(testConfig());
    std::string error;
    require(loaded.load(path.string(), &error),
            "failed to load legacy V4 offset fixture: " + error);
    const auto &descriptor = loaded.entries().front().descriptor;
    require(descriptor.valid(0, 0) == 1 && descriptor.valid(3, 0) == 1,
            "legacy V4 bitset mask changed during height migration");
    require(std::abs(descriptor.values(0, 0) + 0.5) < 1e-12,
            "legacy lower-envelope offset was not removed");
    require(std::abs(descriptor.values(3, 0) - 1.2) < 1e-12,
            "legacy upper-envelope offset was not removed");

    std::error_code ec;
    std::filesystem::remove(path, ec);
}

void testGravityCanonicalizationAndYawRetention()
{
    const double roll = 18.0 * M_PI / 180.0;
    const double yaw = 30.0 * M_PI / 180.0;
    const Eigen::Matrix3d R_world_body =
        Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix() *
        Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX()).toRotationMatrix();
    const Eigen::Vector3d up_body =
        R_world_body.transpose() * Eigen::Vector3d::UnitZ();

    Eigen::Matrix3d R_G_B;
    require(sc::makeGravityCanonicalRotation(up_body, R_G_B),
            "failed to construct gravity-canonical rotation");
    require((R_G_B * up_body - Eigen::Vector3d::UnitZ()).norm() < 1e-12,
            "gravity-canonical rotation did not map up to +Z");

    const Eigen::Matrix3d seed_rotation =
        Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix() * R_G_B;
    require((seed_rotation * up_body - Eigen::Vector3d::UnitZ()).norm() < 1e-12,
            "Rz(yaw) * Rg seed did not preserve gravity alignment");

    sc::PointCloud level_scan;
    level_scan.push_back(point(4.0F, 1.0F, 0.5F));
    level_scan.push_back(point(2.0F, 5.0F, 2.0F));
    level_scan.push_back(point(-3.0F, 2.0F, 3.0F));

    sc::PointCloud tilted_scan;
    for (const auto &level_point : level_scan.points)
    {
        const Eigen::Vector3d p_world(level_point.x, level_point.y, level_point.z);
        const Eigen::Vector3d p_body = R_world_body.transpose() * p_world;
        tilted_scan.push_back(point(
            static_cast<float>(p_body.x()),
            static_cast<float>(p_body.y()),
            static_cast<float>(p_body.z())));
    }
    const sc::PointCloud canonical = sc::gravityCanonicalize(tilted_scan, R_G_B);
    require(canonical.size() == level_scan.size(),
            "gravity canonicalization changed finite point count");

    const Eigen::Matrix3d expected_yaw_only =
        Eigen::AngleAxisd(-yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    for (std::size_t i = 0; i < canonical.size(); ++i)
    {
        const Eigen::Vector3d expected = expected_yaw_only * Eigen::Vector3d(
            level_scan[i].x, level_scan[i].y, level_scan[i].z);
        const Eigen::Vector3d actual(canonical[i].x, canonical[i].y, canonical[i].z);
        require((actual - expected).norm() < 1e-5,
                "gravity canonicalization removed or distorted residual yaw");
    }
}

void testCanonicalYawReconstructsFullSeedRotation()
{
    const Eigen::Matrix3d R_map_body =
        Eigen::AngleAxisd(31.0 * M_PI / 180.0, Eigen::Vector3d::UnitZ()).toRotationMatrix() *
        Eigen::AngleAxisd(-13.0 * M_PI / 180.0, Eigen::Vector3d::UnitY()).toRotationMatrix() *
        Eigen::AngleAxisd(17.0 * M_PI / 180.0, Eigen::Vector3d::UnitX()).toRotationMatrix();
    const Eigen::Vector3d up_body =
        R_map_body.transpose() * Eigen::Vector3d::UnitZ();
    Eigen::Matrix3d R_G_B;
    require(sc::makeGravityCanonicalRotation(up_body, R_G_B),
            "failed to level combined roll/pitch attitude");

    const Eigen::Matrix3d R_map_descriptor = R_map_body * R_G_B.transpose();
    require((R_map_descriptor * Eigen::Vector3d::UnitZ() -
             Eigen::Vector3d::UnitZ()).norm() < 1e-12,
            "map-to-descriptor rotation retained roll/pitch");
    const double canonical_yaw =
        std::atan2(R_map_descriptor(1, 0), R_map_descriptor(0, 0));
    const Eigen::Matrix3d reconstructed =
        Eigen::AngleAxisd(canonical_yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix() * R_G_B;
    require((reconstructed - R_map_body).norm() < 1e-12,
            "canonical yaw and Rg did not reconstruct map<-body seed rotation");
}

void testUpsideDownGravityConvention()
{
    Eigen::Matrix3d rotation;
    require(sc::makeGravityCanonicalRotation(-Eigen::Vector3d::UnitZ(), rotation),
            "upside-down gravity direction was rejected");
    require((rotation * (-Eigen::Vector3d::UnitZ()) -
             Eigen::Vector3d::UnitZ()).norm() < 1e-12,
            "upside-down convention did not map up to +Z");
    const Eigen::Matrix3d expected =
        Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitX()).toRotationMatrix();
    require((rotation - expected).norm() < 1e-12,
            "upside-down convention is not the stable +X 180-degree rotation");
}

void testTextV2MaskCompatibility()
{
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() / "fast_lio_scan_context_mask_v2_compat_test.scd";
    {
        std::ofstream out(path);
        require(static_cast<bool>(out), "failed to create V2 database fixture");
        out << "FAST_LIO_SCAN_CONTEXT_DB_V2\n";
        out << "PARAMS 3 4 9 2 1 1 0.5 0.5 2\n";
        out << "ENTRIES 1\n";
        out << "ENTRY 0 1 0 0 0 0 0 0\n";
        out << "DESC\n";
        for (int row = 0; row < 6; ++row)
        {
            for (int col = 0; col < 4; ++col)
            {
                if (col > 0)
                    out << ' ';
                out << 0;
            }
            out << '\n';
        }
        out << "MASK\n";
        for (int row = 0; row < 6; ++row)
        {
            for (int col = 0; col < 4; ++col)
            {
                if (col > 0)
                    out << ' ';
                out << ((row == 0 && col == 0) ? 1 : 0);
            }
            out << '\n';
        }
        out << "END_ENTRY\n";
    }

    sc::Database loaded(testConfig());
    std::string error;
    require(loaded.load(path.string(), &error), "failed to load V2 database: " + error);
    require(!loaded.legacyMasksInferred(), "V2 database was incorrectly marked as legacy");
    require(loaded.entries().front().descriptor.valid(0, 0) == 1,
            "V2 compatibility path lost an explicit encoded-zero valid bit");
    require(std::abs(loaded.entries().front().descriptor.values(0, 0) + 2.0) < 1e-12,
            "V2 compatibility path did not remove the legacy height offset");
    require(loaded.entries().front().descriptor.valid(0, 1) == 0,
            "V2 compatibility path changed an empty mask cell");

    std::error_code ec;
    std::filesystem::remove(path, ec);
}

void testRingKeyKeepsObservationDensity()
{
    sc::Config config = testConfig();
    config.dual_z_layer_enable = false;
    config.candidate_top_k = 1;
    config.distance_thresh = 1.0;
    sc::Database database(config);

    sc::Descriptor dense;
    dense.values = Eigen::MatrixXd::Zero(3, 4);
    dense.valid = Eigen::Array<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic>::Ones(3, 4);
    for (int row = 0; row < dense.values.rows(); ++row)
        dense.values.row(row).setConstant(static_cast<double>(row + 1));

    sc::Descriptor sparse = dense;
    sparse.valid.setZero();
    sparse.valid.col(0).setOnes();

    sc::Pose sparse_pose;
    sparse_pose.x = -1.0;
    sc::Pose dense_pose;
    dense_pose.x = 1.0;
    database.addEntry(1.0, sparse_pose, sparse);
    database.addEntry(2.0, dense_pose, dense);

    const auto candidates = database.query(dense, false);
    require(candidates.size() == 1, "ring-key density test returned an unexpected candidate count");
    require(candidates.front().index == 1,
            "ring key discarded observation density and selected a sparse alias");
}

void testLegacyMaskInference()
{
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() / "fast_lio_scan_context_mask_v1_test.scd";
    {
        std::ofstream out(path);
        require(static_cast<bool>(out), "failed to create legacy database fixture");
        out << "FAST_LIO_SCAN_CONTEXT_DB_V1\n";
        out << "PARAMS 3 4 9 2 1 1 0.5 0.5\n";
        out << "ENTRIES 1\n";
        out << "ENTRY 0 1 0 0 0 0 0 0\n";
        out << "DESC\n";
        for (int row = 0; row < 6; ++row)
        {
            for (int col = 0; col < 4; ++col)
            {
                if (col > 0)
                    out << ' ';
                out << ((row == 0 && col == 0) ? 1 : 0);
            }
            out << '\n';
        }
        out << "END_ENTRY\n";
    }

    sc::Database loaded(testConfig());
    std::string error;
    require(loaded.load(path.string(), &error), "failed to load V1 database: " + error);
    require(loaded.legacyMasksInferred(), "V1 database did not report inferred masks");
    require(loaded.entries().front().descriptor.valid(0, 0) == 1,
            "V1 nonzero value was not inferred as valid");
    require(loaded.entries().front().descriptor.valid(0, 1) == 0,
            "V1 zero value was not inferred as empty");

    std::error_code ec;
    std::filesystem::remove(path, ec);
}

}  // namespace

int main()
{
    try
    {
        testEnvelopeMasks();
        testPlatformOriginHeightNormalizesDescriptor();
        testDefaultRetrievalHeightOffset();
        testIndependentPlatformHeightsProduceCommonGroundDescriptor();
        testVerticalShiftChangesEnvelopeMasks();
        testYawConditionedStableHeightEstimation();
        testStableHalfRejectsSplitBoundaryResiduals();
        testJointMaskDistance();
        testAbsentDualChannelFallsBackToSupportedEnvelope();
        testAbsentUpperFallbackRequiresAnUpperFreeMap();
        testMixedMapUsesCandidateLocalUpperFallback();
        testAdaptivePhysicalSplit();
        testRetrievalHeightOffsetKeepsSignedDescriptor();
        testBoundaryFilterDoesNotChangeRetrieval();
        testBoundaryCellsDoNotBiasVerticalShift();
        testSectorSupportRejectsSparseAlias();
        testV7BitsetDatabaseMaskRoundTrip();
        testV6SignedHeightMigration();
        testLegacyV4HeightOffsetMigration();
        testTextV2MaskCompatibility();
        testRingKeyKeepsObservationDensity();
        testLegacyMaskInference();
        testGravityCanonicalizationAndYawRetention();
        testCanonicalYawReconstructsFullSeedRotation();
        testUpsideDownGravityConvention();
    }
    catch (const std::exception &error)
    {
        std::cerr << "scan_context_mask_test failed: " << error.what() << '\n';
        return 1;
    }

    std::cout << "scan_context_mask_test passed\n";
    return 0;
}
