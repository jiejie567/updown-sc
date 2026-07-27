#include "source_ray_export.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#include <Eigen/Geometry>
#include <sensor_msgs/msg/point_field.hpp>

#include "pcd_save_utils.hpp"

namespace fast_lio::source_ray_export
{
namespace
{

constexpr std::array<char, 8> kRayMagic{{'S', 'R', 'R', 'A', 'Y', 'S', '1', '\0'}};

std::vector<std::string> splitCsv(const std::string &line)
{
    std::vector<std::string> fields;
    std::stringstream stream(line);
    std::string field;
    while (std::getline(stream, field, ','))
    {
        if (!field.empty() && field.back() == '\r')
            field.pop_back();
        fields.push_back(field);
    }
    return fields;
}

template <typename T>
bool readScalar(
    const sensor_msgs::msg::PointCloud2 &message,
    std::size_t point_offset,
    std::size_t field_offset,
    T &value)
{
    const std::size_t offset = point_offset + field_offset;
    if (offset + sizeof(T) > message.data.size())
        return false;
    std::memcpy(&value, message.data.data() + offset, sizeof(T));
    return true;
}

const sensor_msgs::msg::PointField *findField(
    const sensor_msgs::msg::PointCloud2 &message,
    const std::string &name)
{
    const auto it = std::find_if(
        message.fields.begin(), message.fields.end(),
        [&name](const sensor_msgs::msg::PointField &field)
        {
            return field.name == name;
        });
    return it == message.fields.end() ? nullptr : &*it;
}

Eigen::Matrix4d tumPose(
    double x,
    double y,
    double z,
    double qx,
    double qy,
    double qz,
    double qw)
{
    Eigen::Quaterniond quaternion(qw, qx, qy, qz);
    const double norm = quaternion.norm();
    if (!std::isfinite(norm) || norm < 1.0e-12)
        throw std::runtime_error("invalid quaternion in PGO TUM");
    quaternion.normalize();
    Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
    transform.block<3, 3>(0, 0) = quaternion.toRotationMatrix();
    transform.block<3, 1>(0, 3) = Eigen::Vector3d(x, y, z);
    return transform;
}

struct VoxelKey
{
    int64_t x = 0;
    int64_t y = 0;
    int64_t z = 0;
    uint8_t sensor = 0;

    bool operator==(const VoxelKey &other) const
    {
        return x == other.x && y == other.y && z == other.z &&
               sensor == other.sensor;
    }
};

struct VoxelKeyHash
{
    std::size_t operator()(const VoxelKey &key) const
    {
        std::size_t seed = std::hash<int64_t>{}(key.x);
        seed ^= std::hash<int64_t>{}(key.y) + 0x9e3779b9U + (seed << 6U) +
                (seed >> 2U);
        seed ^= std::hash<int64_t>{}(key.z) + 0x9e3779b9U + (seed << 6U) +
                (seed >> 2U);
        seed ^= std::hash<unsigned int>{}(key.sensor) + 0x9e3779b9U +
                (seed << 6U) + (seed >> 2U);
        return seed;
    }
};

struct WorldRay
{
    Eigen::Vector3d origin = Eigen::Vector3d::Zero();
    Eigen::Vector3d hit = Eigen::Vector3d::Zero();
    uint8_t sensor = 0;
    uint32_t original_index = 0;
};

struct VoxelChoice
{
    std::size_t ray_index = 0;
    double center_distance_sq = std::numeric_limits<double>::infinity();
    uint32_t original_index = std::numeric_limits<uint32_t>::max();
};

bool writeLittleEndian(std::ofstream &stream, uint32_t value)
{
    stream.write(reinterpret_cast<const char *>(&value), sizeof(value));
    return static_cast<bool>(stream);
}

bool writeLittleEndian(std::ofstream &stream, uint64_t value)
{
    stream.write(reinterpret_cast<const char *>(&value), sizeof(value));
    return static_cast<bool>(stream);
}

std::string targetStem(int target_index)
{
    std::ostringstream stream;
    stream << std::setfill('0') << std::setw(6) << target_index;
    return stream.str();
}

}  // namespace

Exporter::~Exporter()
{
    if (configured_ && config_.enable && !finalized_)
    {
        std::string ignored;
        finalize(ignored);
    }
}

bool Exporter::configure(const Config &config, std::string &error)
{
    error.clear();
    configured_ = false;
    config_ = config;
    finalized_ = false;
    exported_count_ = 0;
    total_eligible_count_ = 0;
    total_written_count_ = 0;
    maximum_ray_length_error_m_ = 0.0;
    maximum_p99_ray_length_error_m_ = 0.0;
    frames_.clear();
    active_pose_indices_.clear();
    exported_pose_indices_.clear();

    if (!config_.enable)
    {
        configured_ = true;
        return true;
    }
    if (config_.manifest_csv.empty() || config_.pgo_tum.empty() ||
        config_.output_dir.empty())
    {
        error = "manifest_csv, pgo_tum, and output_dir are required";
        return false;
    }
    if (config_.expected_frame_count <= 0)
    {
        error = "expected_frame_count must be positive";
        return false;
    }
    if (!std::isfinite(config_.timestamp_tolerance_us) ||
        !std::isfinite(config_.blind_radius_m) ||
        !std::isfinite(config_.blind_z_min_m) ||
        !std::isfinite(config_.blind_z_max_m) ||
        !std::isfinite(config_.maximum_range_m) ||
        !std::isfinite(config_.endpoint_voxel_m) ||
        config_.timestamp_tolerance_us < 0.0 ||
        config_.blind_radius_m < 0.0 ||
        config_.blind_z_min_m > config_.blind_z_max_m ||
        config_.maximum_range_m <= 0.0 ||
        config_.endpoint_voxel_m < 0.0 ||
        config_.scan_line <= 0 || config_.scan_line > 256)
    {
        error = "invalid source-ray numeric configuration";
        return false;
    }

    std::ifstream tum_stream(config_.pgo_tum);
    if (!tum_stream)
    {
        error = "failed to open PGO TUM: " + config_.pgo_tum;
        return false;
    }
    struct TumRow
    {
        int64_t timestamp_ns = 0;
        Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
    };
    std::vector<TumRow, Eigen::aligned_allocator<TumRow>> tum_rows;
    std::string line;
    while (std::getline(tum_stream, line))
    {
        if (line.empty() || line[0] == '#')
            continue;
        std::istringstream row(line);
        double stamp = 0.0;
        double x = 0.0, y = 0.0, z = 0.0;
        double qx = 0.0, qy = 0.0, qz = 0.0, qw = 1.0;
        if (!(row >> stamp >> x >> y >> z >> qx >> qy >> qz >> qw))
        {
            error = "malformed PGO TUM row";
            return false;
        }
        if (!std::isfinite(stamp) || !std::isfinite(x) ||
            !std::isfinite(y) || !std::isfinite(z) ||
            !std::isfinite(qx) || !std::isfinite(qy) ||
            !std::isfinite(qz) || !std::isfinite(qw))
        {
            error = "PGO TUM contains a non-finite value";
            return false;
        }
        TumRow parsed;
        parsed.timestamp_ns = static_cast<int64_t>(std::llround(stamp * 1.0e9));
        try
        {
            parsed.transform = tumPose(x, y, z, qx, qy, qz, qw);
        }
        catch (const std::exception &exception)
        {
            error = exception.what();
            return false;
        }
        tum_rows.push_back(parsed);
    }
    if (static_cast<int>(tum_rows.size()) != config_.expected_frame_count)
    {
        error =
            "PGO TUM count mismatch: expected " +
            std::to_string(config_.expected_frame_count) + ", got " +
            std::to_string(tum_rows.size());
        return false;
    }
    std::sort(
        tum_rows.begin(), tum_rows.end(),
        [](const TumRow &lhs, const TumRow &rhs)
        {
            return lhs.timestamp_ns < rhs.timestamp_ns;
        });
    for (std::size_t index = 1; index < tum_rows.size(); ++index)
    {
        if (tum_rows[index - 1].timestamp_ns >= tum_rows[index].timestamp_ns)
        {
            error = "PGO TUM timestamps must be unique";
            return false;
        }
    }

    std::ifstream csv_stream(config_.manifest_csv);
    if (!csv_stream)
    {
        error = "failed to open block manifest: " + config_.manifest_csv;
        return false;
    }
    if (!std::getline(csv_stream, line))
    {
        error = "empty block manifest";
        return false;
    }
    const std::vector<std::string> header = splitCsv(line);
    const std::vector<std::string> expected_header{
        "target_index",
        "pose_timestamp_ns",
        "frame_reference_timestamp_ns",
        "pose_delta_ns",
        "front_block_index",
        "back_block_index",
        "boundary_index",
        "front_point_count",
        "back_point_count"};
    std::vector<std::string> legacy_header = expected_header;
    legacy_header[0] = "pgo_index";
    if (header != expected_header && header != legacy_header)
    {
        error =
            "unexpected block manifest header (expected target_index; "
            "legacy pgo_index is also accepted)";
        return false;
    }

    std::unordered_set<int> seen_indices;
    std::unordered_set<std::size_t> seen_tum_rows;
    const int64_t tolerance_ns = static_cast<int64_t>(
        std::llround(config_.timestamp_tolerance_us * 1000.0));
    while (std::getline(csv_stream, line))
    {
        if (line.empty())
            continue;
        const std::vector<std::string> fields = splitCsv(line);
        if (fields.size() != expected_header.size())
        {
            error = "malformed block manifest row";
            return false;
        }
        SelectedFrame frame;
        try
        {
            frame.pose_index = std::stoi(fields[0]);
            frame.pose_timestamp_ns = std::stoll(fields[1]);
            frame.frame_reference_timestamp_ns = std::stoll(fields[2]);
            const int64_t pose_delta_ns = std::stoll(fields[3]);
            frame.front_block_index = std::stoi(fields[4]);
            frame.back_block_index = std::stoi(fields[5]);
            frame.boundary_index =
                static_cast<std::size_t>(std::stoull(fields[6]));
            frame.front_point_count =
                static_cast<std::size_t>(std::stoull(fields[7]));
            frame.back_point_count =
                static_cast<std::size_t>(std::stoull(fields[8]));
            if (std::llabs(pose_delta_ns) > tolerance_ns)
            {
                error = "block manifest pose timestamp exceeds tolerance";
                return false;
            }
            if (frame.frame_reference_timestamp_ns -
                    frame.pose_timestamp_ns !=
                pose_delta_ns)
            {
                error =
                    "block manifest pose_delta_ns is inconsistent with its "
                    "two timestamp columns";
                return false;
            }
        }
        catch (const std::exception &)
        {
            error = "non-numeric block manifest row";
            return false;
        }
        if (frame.pose_index < 0 ||
            !seen_indices.insert(frame.pose_index).second)
        {
            error = "invalid or duplicate target index in block manifest";
            return false;
        }
        const bool dual_sensor =
            (frame.front_block_index == 0 || frame.front_block_index == 1) &&
            frame.back_block_index == 1 - frame.front_block_index &&
            frame.front_point_count > 0 && frame.back_point_count > 0;
        const bool front_only =
            frame.front_block_index == 0 && frame.back_block_index == -1 &&
            frame.front_point_count > 0 && frame.back_point_count == 0;
        if (!dual_sensor && !front_only)
        {
            error =
                "invalid sensor layout in manifest; expected two concatenated "
                "front/back blocks or one front-only block";
            return false;
        }
        const auto tum_it = std::lower_bound(
            tum_rows.begin(), tum_rows.end(), frame.pose_timestamp_ns,
            [](const TumRow &row, int64_t stamp)
            {
                return row.timestamp_ns < stamp;
            });
        const TumRow *tum = nullptr;
        std::size_t tum_index = 0;
        int64_t best_delta = std::numeric_limits<int64_t>::max();
        if (tum_it != tum_rows.end())
        {
            tum = &*tum_it;
            tum_index = static_cast<std::size_t>(
                std::distance(tum_rows.begin(), tum_it));
            best_delta = std::llabs(tum->timestamp_ns - frame.pose_timestamp_ns);
        }
        if (tum_it != tum_rows.begin())
        {
            const auto previous = tum_it - 1;
            const int64_t delta =
                std::llabs(previous->timestamp_ns - frame.pose_timestamp_ns);
            if (delta < best_delta)
            {
                tum = &*previous;
                tum_index = static_cast<std::size_t>(
                    std::distance(tum_rows.begin(), previous));
                best_delta = delta;
            }
        }
        if (tum == nullptr || best_delta > tolerance_ns)
        {
            error = "PGO TUM and block manifest timestamps disagree";
            return false;
        }
        if (!seen_tum_rows.insert(tum_index).second)
        {
            error = "multiple manifest rows matched the same PGO TUM pose";
            return false;
        }
        if (frame.pose_index != static_cast<int>(tum_index))
        {
            error =
                "target_index does not match its timestamp-sorted PGO TUM "
                "row";
            return false;
        }
        frame.T_world_base = tum->transform;
        frames_.push_back(frame);
    }
    if (static_cast<int>(frames_.size()) != config_.expected_frame_count)
    {
        error =
            "block manifest count mismatch: expected " +
            std::to_string(config_.expected_frame_count) + ", got " +
            std::to_string(frames_.size());
        return false;
    }
    std::sort(
        frames_.begin(), frames_.end(),
        [](const SelectedFrame &lhs, const SelectedFrame &rhs)
        {
            return lhs.frame_reference_timestamp_ns <
                   rhs.frame_reference_timestamp_ns;
        });
    for (std::size_t index = 1; index < frames_.size(); ++index)
    {
        if (frames_[index - 1].frame_reference_timestamp_ns >=
            frames_[index].frame_reference_timestamp_ns)
        {
            error = "block manifest frame timestamps are not unique";
            return false;
        }
    }
    for (std::size_t index = 0; index < frames_.size(); ++index)
    {
        if (frames_[index].pose_index != static_cast<int>(index))
        {
            error =
                "target_index must equal the timestamp-sorted pose row "
                "(0..expected_frame_count-1)";
            return false;
        }
    }

    if (config_.selected_pose_indices.empty())
    {
        for (const SelectedFrame &frame : frames_)
            active_pose_indices_.insert(frame.pose_index);
    }
    else
    {
        for (const int64_t pose_index : config_.selected_pose_indices)
        {
            if (pose_index < 0 ||
                seen_indices.count(static_cast<int>(pose_index)) == 0)
            {
                error = "selected target index is absent from the manifest";
                return false;
            }
            active_pose_indices_.insert(static_cast<int>(pose_index));
        }
    }

    const std::filesystem::path output_root(config_.output_dir);
    std::error_code filesystem_error;
    if (std::filesystem::exists(output_root) && !config_.overwrite)
    {
        error = "source-ray output already exists: " + output_root.string();
        return false;
    }
    if (config_.overwrite)
    {
        std::filesystem::remove_all(output_root, filesystem_error);
        if (filesystem_error)
        {
            error = "failed to clear source-ray output: " +
                    filesystem_error.message();
            return false;
        }
    }
    std::filesystem::create_directories(
        output_root / "srrays", filesystem_error);
    if (filesystem_error)
    {
        error = "failed to create source-ray output: " +
                filesystem_error.message();
        return false;
    }
    if (config_.save_sensor_pcd)
    {
        std::filesystem::create_directories(
            output_root / "frame_end_base" / "front", filesystem_error);
        if (!filesystem_error)
        {
            std::filesystem::create_directories(
                output_root / "frame_end_base" / "back", filesystem_error);
        }
        if (filesystem_error)
        {
            error = "failed to create frame-end sensor PCD output: " +
                    filesystem_error.message();
            return false;
        }
    }
    manifest_stream_.open(output_root / "export_manifest.csv", std::ios::trunc);
    if (!manifest_stream_)
    {
        error = "failed to create source-ray export manifest";
        return false;
    }
    manifest_stream_
        << "target_index,frame_reference_timestamp_ns,frame_end_timestamp_ns,"
           "fused_points,eligible_front,eligible_back,eligible_total,"
           "written_front,written_back,written_total,"
           "ray_length_error_max_m,ray_length_error_p99_m,srrays_path\n";
    local_cloud_manifest_stream_.open(
        output_root / "local_cloud_manifest.csv", std::ios::trunc);
    if (!local_cloud_manifest_stream_)
    {
        error = "failed to create frame-end local-cloud manifest";
        return false;
    }
    local_cloud_manifest_stream_
        << "target_index,pose_timestamp_ns,message_header_timestamp_ns,"
           "frame_reference_timestamp_ns,frame_end_timestamp_ns,"
           "frame_duration_ns,local_frame,gravity_up_base_x,"
           "gravity_up_base_y,gravity_up_base_z,eligible_front,"
           "eligible_back,front_pcd_path,back_pcd_path\n";
    configured_ = true;
    return true;
}

const Exporter::SelectedFrame *Exporter::findSelectedFrame(
    int64_t reference_ns) const
{
    const auto it = std::lower_bound(
        frames_.begin(), frames_.end(), reference_ns,
        [](const SelectedFrame &frame, int64_t timestamp)
        {
            return frame.frame_reference_timestamp_ns < timestamp;
        });
    const int64_t tolerance_ns = static_cast<int64_t>(
        std::llround(config_.timestamp_tolerance_us * 1000.0));
    const SelectedFrame *best = nullptr;
    int64_t best_delta = std::numeric_limits<int64_t>::max();
    if (it != frames_.end())
    {
        const int64_t delta =
            std::llabs(it->frame_reference_timestamp_ns - reference_ns);
        best = &*it;
        best_delta = delta;
    }
    if (it != frames_.begin())
    {
        const auto previous = it - 1;
        const int64_t delta =
            std::llabs(previous->frame_reference_timestamp_ns - reference_ns);
        if (delta < best_delta)
        {
            best = &*previous;
            best_delta = delta;
        }
    }
    return best != nullptr && best_delta <= tolerance_ns ? best : nullptr;
}

std::unique_ptr<PendingFrame> Exporter::prepare(
    const sensor_msgs::msg::PointCloud2 &message,
    std::string &error) const
{
    error.clear();
    if (!config_.enable)
        return nullptr;
    if (message.is_bigendian)
    {
        error = "big-endian PointCloud2 is unsupported";
        return nullptr;
    }

    const auto *x_field = findField(message, "x");
    const auto *y_field = findField(message, "y");
    const auto *z_field = findField(message, "z");
    const auto *intensity_field = findField(message, "intensity");
    const auto *tag_field = findField(message, "tag");
    const auto *line_field = findField(message, "line");
    const auto *timestamp_field = findField(message, "timestamp");
    if (!x_field || !y_field || !z_field || !intensity_field || !tag_field ||
        !line_field || !timestamp_field)
    {
        error = "PointCloud2 misses x/y/z/intensity/tag/line/timestamp";
        return nullptr;
    }
    if (x_field->datatype != sensor_msgs::msg::PointField::FLOAT32 ||
        y_field->datatype != sensor_msgs::msg::PointField::FLOAT32 ||
        z_field->datatype != sensor_msgs::msg::PointField::FLOAT32 ||
        intensity_field->datatype != sensor_msgs::msg::PointField::FLOAT32 ||
        tag_field->datatype != sensor_msgs::msg::PointField::UINT8 ||
        line_field->datatype != sensor_msgs::msg::PointField::UINT8 ||
        timestamp_field->datatype != sensor_msgs::msg::PointField::FLOAT64)
    {
        error = "PointCloud2 source-ray field types do not match MID360";
        return nullptr;
    }
    const std::size_t width = static_cast<std::size_t>(message.width);
    const std::size_t height = static_cast<std::size_t>(message.height);
    const std::size_t point_step = static_cast<std::size_t>(message.point_step);
    const auto field_fits = [point_step](
                                const sensor_msgs::msg::PointField *field,
                                std::size_t width_bytes)
    {
        const std::size_t offset = static_cast<std::size_t>(field->offset);
        return offset <= point_step && width_bytes <= point_step - offset;
    };
    if (!field_fits(x_field, sizeof(float)) ||
        !field_fits(y_field, sizeof(float)) ||
        !field_fits(z_field, sizeof(float)) ||
        !field_fits(intensity_field, sizeof(float)) ||
        !field_fits(tag_field, sizeof(uint8_t)) ||
        !field_fits(line_field, sizeof(uint8_t)) ||
        !field_fits(timestamp_field, sizeof(double)))
    {
        error = "PointCloud2 field extends beyond point_step";
        return nullptr;
    }
    if ((height != 0 &&
         width > std::numeric_limits<std::size_t>::max() / height) ||
        (point_step != 0 &&
         width > std::numeric_limits<std::size_t>::max() / point_step))
    {
        error = "PointCloud2 dimensions overflow size_t";
        return nullptr;
    }
    const std::size_t point_count = width * height;
    const std::size_t packed_row_step = width * point_step;
    const std::size_t row_step =
        message.row_step > 0 ? static_cast<std::size_t>(message.row_step)
                             : packed_row_step;
    if (point_count == 0 || point_step == 0 || row_step < packed_row_step ||
        (height > 0 && row_step > message.data.size() / height))
    {
        error = "invalid PointCloud2 dimensions";
        return nullptr;
    }

    std::vector<int64_t> timestamps(point_count, 0);
    int64_t reference_ns = std::numeric_limits<int64_t>::max();
    int64_t end_ns = std::numeric_limits<int64_t>::min();
    std::size_t linear_index = 0;
    for (std::size_t row = 0; row < height; ++row)
    {
        for (std::size_t column = 0; column < width; ++column, ++linear_index)
        {
            const std::size_t point_offset =
                row * row_step + column * point_step;
            double timestamp = 0.0;
            if (!readScalar(
                    message, point_offset, timestamp_field->offset, timestamp) ||
                !std::isfinite(timestamp))
            {
                error = "invalid native point timestamp";
                return nullptr;
            }
            const int64_t timestamp_ns =
                static_cast<int64_t>(std::llround(timestamp));
            timestamps[linear_index] = timestamp_ns;
            reference_ns = std::min(reference_ns, timestamp_ns);
            end_ns = std::max(end_ns, timestamp_ns);
        }
    }
    const SelectedFrame *selected = findSelectedFrame(reference_ns);
    if (!selected ||
        active_pose_indices_.count(selected->pose_index) == 0)
    {
        return nullptr;
    }

    std::vector<std::size_t> rollback_indices;
    for (std::size_t index = 1; index < timestamps.size(); ++index)
    {
        if (timestamps[index] - timestamps[index - 1] < -1000000LL)
            rollback_indices.push_back(index);
    }
    const bool dual_sensor = selected->back_block_index >= 0;
    if ((dual_sensor && rollback_indices.size() != 1) ||
        (!dual_sensor && !rollback_indices.empty()))
    {
        error = dual_sensor
                    ? "selected dual-sensor cloud does not contain exactly "
                      "one packet rollback"
                    : "selected front-only cloud unexpectedly contains a "
                      "packet rollback";
        return nullptr;
    }
    const std::size_t boundary =
        dual_sensor ? rollback_indices.front() : point_count;
    if (boundary != selected->boundary_index)
    {
        error = "selected fused cloud boundary disagrees with identity manifest";
        return nullptr;
    }
    const std::array<std::size_t, 2> block_counts{{
        boundary,
        point_count - boundary}};
    const std::size_t observed_front_count =
        block_counts[static_cast<std::size_t>(selected->front_block_index)];
    const std::size_t observed_back_count =
        dual_sensor
            ? block_counts[static_cast<std::size_t>(selected->back_block_index)]
            : 0;
    if (observed_front_count != selected->front_point_count ||
        observed_back_count != selected->back_point_count)
    {
        error = "selected fused block counts disagree with identity manifest";
        return nullptr;
    }

    struct RawRay
    {
        PointType hit;
        PointType origin;
        float original_length_m = 0.0F;
        uint8_t sensor = 0;
        uint32_t original_index = 0;
    };
    std::vector<RawRay> rays;
    rays.reserve(point_count);
    const std::array<Eigen::Matrix4d, 2> transforms{{
        config_.T_input_front,
        config_.T_input_back}};
    linear_index = 0;
    for (std::size_t row = 0; row < height; ++row)
    {
        for (std::size_t column = 0; column < width; ++column, ++linear_index)
        {
            const std::size_t point_offset =
                row * row_step + column * point_step;
            float x = 0.0F, y = 0.0F, z = 0.0F, intensity = 0.0F;
            uint8_t tag = 0, line_id = 0;
            if (!readScalar(message, point_offset, x_field->offset, x) ||
                !readScalar(message, point_offset, y_field->offset, y) ||
                !readScalar(message, point_offset, z_field->offset, z) ||
                !readScalar(
                    message, point_offset, intensity_field->offset, intensity) ||
                !readScalar(message, point_offset, tag_field->offset, tag) ||
                !readScalar(
                    message, point_offset, line_field->offset, line_id))
            {
                error = "selected fused cloud field read exceeded message bounds";
                return nullptr;
            }
            if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z) ||
                (std::abs(x) + std::abs(y) + std::abs(z)) < 1.0e-9F ||
                (tag & 0x3FU) != 0U ||
                static_cast<int>(line_id) >= config_.scan_line)
            {
                continue;
            }
            const int block = linear_index < boundary ? 0 : 1;
            const uint8_t sensor =
                static_cast<uint8_t>(
                    !dual_sensor ||
                            block == selected->front_block_index
                        ? 0
                        : 1);
            const Eigen::Matrix4d &T_input_sensor = transforms[sensor];
            const Eigen::Vector3d hit_input(x, y, z);
            const Eigen::Vector3d hit_sensor =
                T_input_sensor.block<3, 3>(0, 0).transpose() *
                (hit_input - T_input_sensor.block<3, 1>(0, 3));
            const double xy_sq =
                hit_sensor.x() * hit_sensor.x() +
                hit_sensor.y() * hit_sensor.y();
            const double range_sq = xy_sq + hit_sensor.z() * hit_sensor.z();
            const bool inside_blind =
                xy_sq <= config_.blind_radius_m * config_.blind_radius_m &&
                hit_sensor.z() >= config_.blind_z_min_m &&
                hit_sensor.z() <= config_.blind_z_max_m;
            if (inside_blind ||
                range_sq >=
                    config_.maximum_range_m * config_.maximum_range_m)
            {
                continue;
            }

            RawRay ray{};
            ray.hit.x = x;
            ray.hit.y = y;
            ray.hit.z = z;
            ray.hit.intensity = intensity;
            ray.hit.normal_x = 0.0F;
            ray.hit.normal_y = 0.0F;
            ray.hit.normal_z = 0.0F;
            ray.hit.curvature = static_cast<float>(
                (timestamps[linear_index] - reference_ns) * 1.0e-6);
            const Eigen::Vector3d origin =
                T_input_sensor.block<3, 1>(0, 3);
            ray.origin.x = static_cast<float>(origin.x());
            ray.origin.y = static_cast<float>(origin.y());
            ray.origin.z = static_cast<float>(origin.z());
            ray.origin.intensity = intensity;
            ray.origin.normal_x = 0.0F;
            ray.origin.normal_y = 0.0F;
            ray.origin.normal_z = 0.0F;
            ray.origin.curvature = ray.hit.curvature;
            ray.original_length_m =
                static_cast<float>((hit_input - origin).norm());
            ray.sensor = sensor;
            ray.original_index = static_cast<uint32_t>(linear_index);
            rays.push_back(ray);
        }
    }
    std::sort(
        rays.begin(), rays.end(),
        [](const RawRay &lhs, const RawRay &rhs)
        {
            if (lhs.hit.curvature != rhs.hit.curvature)
                return lhs.hit.curvature < rhs.hit.curvature;
            return lhs.original_index < rhs.original_index;
        });

    auto pending = std::make_unique<PendingFrame>();
    pending->pose_index = selected->pose_index;
    pending->pose_timestamp_ns = selected->pose_timestamp_ns;
    pending->message_header_timestamp_ns =
        static_cast<int64_t>(message.header.stamp.sec) * 1000000000LL +
        static_cast<int64_t>(message.header.stamp.nanosec);
    pending->frame_reference_timestamp_ns = reference_ns;
    pending->frame_end_timestamp_ns = end_ns;
    pending->front_block_index = selected->front_block_index;
    pending->fused_point_count = point_count;
    pending->eligible_count = rays.size();
    pending->T_world_base = selected->T_world_base;
    pending->hits_input.reserve(rays.size());
    pending->origins_input.reserve(rays.size());
    pending->original_ray_lengths_m.reserve(rays.size());
    pending->sensor_ids.reserve(rays.size());
    pending->original_indices.reserve(rays.size());
    for (const RawRay &ray : rays)
    {
        pending->hits_input.push_back(ray.hit);
        pending->origins_input.push_back(ray.origin);
        pending->original_ray_lengths_m.push_back(ray.original_length_m);
        pending->sensor_ids.push_back(ray.sensor);
        pending->original_indices.push_back(ray.original_index);
        ++pending->eligible_by_sensor[ray.sensor];
    }
    pending->hits_input.width =
        static_cast<uint32_t>(pending->hits_input.size());
    pending->hits_input.height = 1;
    pending->origins_input.width =
        static_cast<uint32_t>(pending->origins_input.size());
    pending->origins_input.height = 1;
    return pending;
}

bool Exporter::writeDeskewed(
    PendingFrame &frame,
    const Eigen::Matrix3d &R_base_input,
    const Eigen::Vector3d &t_base_input,
    const Eigen::Vector3d &gravity_up_base,
    std::string &error)
{
    error.clear();
    if (!config_.enable)
        return true;
    if (frame.hits_input.size() != frame.origins_input.size() ||
        frame.hits_input.size() != frame.original_ray_lengths_m.size() ||
        frame.hits_input.size() != frame.sensor_ids.size() ||
        frame.hits_input.size() != frame.original_indices.size() ||
        frame.hits_input.size() != frame.eligible_count)
    {
        error = "deskewed source-ray arrays are not aligned";
        return false;
    }
    if (!R_base_input.allFinite() || !t_base_input.allFinite() ||
        !gravity_up_base.allFinite() ||
        gravity_up_base.squaredNorm() < 1.0e-12)
    {
        error = "invalid frame-end base transform or gravity direction";
        return false;
    }
    if (exported_pose_indices_.count(frame.pose_index) != 0)
    {
        error =
            "target was exported more than once: " +
            std::to_string(frame.pose_index);
        return false;
    }

    std::vector<WorldRay> rays;
    rays.reserve(frame.eligible_count);
    std::vector<double> ray_length_errors;
    ray_length_errors.reserve(frame.eligible_count);
    std::array<PointCloudXYZI::Ptr, 2> local_clouds{{
        PointCloudXYZI::Ptr(new PointCloudXYZI()),
        PointCloudXYZI::Ptr(new PointCloudXYZI())}};
    local_clouds[0]->reserve(frame.eligible_by_sensor[0]);
    local_clouds[1]->reserve(frame.eligible_by_sensor[1]);
    const Eigen::Matrix3d R_world_base =
        frame.T_world_base.block<3, 3>(0, 0);
    const Eigen::Vector3d t_world_base =
        frame.T_world_base.block<3, 1>(0, 3);
    for (std::size_t index = 0; index < frame.eligible_count; ++index)
    {
        const auto &hit = frame.hits_input[index];
        const auto &origin = frame.origins_input[index];
        const Eigen::Vector3d hit_input(hit.x, hit.y, hit.z);
        const Eigen::Vector3d origin_input(origin.x, origin.y, origin.z);
        ray_length_errors.push_back(
            std::abs(
                (hit_input - origin_input).norm() -
                static_cast<double>(frame.original_ray_lengths_m[index])));
        const Eigen::Vector3d hit_base =
            R_base_input * hit_input + t_base_input;
        const Eigen::Vector3d origin_base =
            R_base_input * origin_input + t_base_input;
        PointType local_point = hit;
        local_point.x = static_cast<float>(hit_base.x());
        local_point.y = static_cast<float>(hit_base.y());
        local_point.z = static_cast<float>(hit_base.z());
        local_clouds[frame.sensor_ids[index]]->push_back(local_point);
        WorldRay ray;
        ray.hit = R_world_base * hit_base + t_world_base;
        ray.origin = R_world_base * origin_base + t_world_base;
        ray.sensor = frame.sensor_ids[index];
        ray.original_index = frame.original_indices[index];
        if (!ray.hit.allFinite() || !ray.origin.allFinite())
        {
            error = "deskewed source ray contains a non-finite coordinate";
            return false;
        }
        rays.push_back(ray);
    }
    std::sort(ray_length_errors.begin(), ray_length_errors.end());
    const double maximum_ray_length_error_m =
        ray_length_errors.empty() ? 0.0 : ray_length_errors.back();
    const std::size_t p99_index =
        ray_length_errors.empty()
            ? 0
            : std::min(
                  ray_length_errors.size() - 1,
                  static_cast<std::size_t>(
                      std::ceil(0.99 * ray_length_errors.size())) -
                      1);
    const double p99_ray_length_error_m =
        ray_length_errors.empty() ? 0.0 : ray_length_errors[p99_index];
    if (maximum_ray_length_error_m > 1.0e-4 ||
        p99_ray_length_error_m > 1.0e-5)
    {
        std::ostringstream message;
        message
            << "hit/origin deskew did not preserve ray length: max="
            << std::setprecision(12) << maximum_ray_length_error_m
            << " m p99=" << p99_ray_length_error_m << " m";
        error = message.str();
        return false;
    }
    for (auto &cloud : local_clouds)
    {
        cloud->width = static_cast<uint32_t>(cloud->size());
        cloud->height = 1;
        cloud->is_dense = true;
    }

    std::string front_pcd_path;
    std::string back_pcd_path;
    if (config_.save_sensor_pcd)
    {
        const std::string stem = targetStem(frame.pose_index);
        front_pcd_path =
            (std::filesystem::path(config_.output_dir) / "frame_end_base" /
             "front" / (stem + ".pcd"))
                .string();
        back_pcd_path =
            (std::filesystem::path(config_.output_dir) / "frame_end_base" /
             "back" / (stem + ".pcd"))
                .string();
        const auto front_save =
            pcd_save::writeBinary(front_pcd_path, local_clouds[0], 0.0);
        if (!front_save.success ||
            front_save.output_points != frame.eligible_by_sensor[0])
        {
            error =
                "failed to save exact front frame-end PCD: " +
                front_save.error;
            return false;
        }
        const auto back_save =
            pcd_save::writeBinary(back_pcd_path, local_clouds[1], 0.0);
        if (!back_save.success ||
            back_save.output_points != frame.eligible_by_sensor[1])
        {
            error =
                "failed to save exact back frame-end PCD: " +
                back_save.error;
            return false;
        }
    }

    std::vector<std::size_t> selected_indices;
    if (config_.endpoint_voxel_m <= 0.0)
    {
        selected_indices.resize(rays.size());
        for (std::size_t index = 0; index < rays.size(); ++index)
            selected_indices[index] = index;
    }
    else
    {
        const double voxel = config_.endpoint_voxel_m;
        std::unordered_map<VoxelKey, VoxelChoice, VoxelKeyHash> choices;
        choices.reserve(rays.size());
        for (std::size_t index = 0; index < rays.size(); ++index)
        {
            const WorldRay &ray = rays[index];
            VoxelKey key;
            key.x = static_cast<int64_t>(std::floor(ray.hit.x() / voxel));
            key.y = static_cast<int64_t>(std::floor(ray.hit.y() / voxel));
            key.z = static_cast<int64_t>(std::floor(ray.hit.z() / voxel));
            key.sensor = ray.sensor;
            const Eigen::Vector3d center(
                (static_cast<double>(key.x) + 0.5) * voxel,
                (static_cast<double>(key.y) + 0.5) * voxel,
                (static_cast<double>(key.z) + 0.5) * voxel);
            const double distance_sq = (ray.hit - center).squaredNorm();
            auto [choice_it, inserted] =
                choices.emplace(key, VoxelChoice{index, distance_sq, ray.original_index});
            if (!inserted)
            {
                VoxelChoice &choice = choice_it->second;
                if (distance_sq < choice.center_distance_sq ||
                    (distance_sq == choice.center_distance_sq &&
                     ray.original_index < choice.original_index))
                {
                    choice =
                        VoxelChoice{index, distance_sq, ray.original_index};
                }
            }
        }
        selected_indices.reserve(choices.size());
        for (const auto &entry : choices)
            selected_indices.push_back(entry.second.ray_index);
        std::sort(
            selected_indices.begin(), selected_indices.end(),
            [&rays](std::size_t lhs, std::size_t rhs)
            {
                return rays[lhs].original_index < rays[rhs].original_index;
            });
    }

    const std::filesystem::path output_path =
        std::filesystem::path(config_.output_dir) / "srrays" /
        (targetStem(frame.pose_index) + ".srrays");
    if (std::filesystem::exists(output_path) && !config_.overwrite)
    {
        error = "source-ray shard already exists: " + output_path.string();
        return false;
    }
    std::ofstream stream(output_path, std::ios::binary | std::ios::trunc);
    if (!stream)
    {
        error = "failed to create source-ray shard: " + output_path.string();
        return false;
    }
    stream.write(kRayMagic.data(), static_cast<std::streamsize>(kRayMagic.size()));
    if (!writeLittleEndian(stream, uint32_t{1}) ||
        !writeLittleEndian(stream, uint32_t{0}) ||
        !writeLittleEndian(
            stream, static_cast<uint64_t>(selected_indices.size())))
    {
        error = "failed to write source-ray shard header";
        return false;
    }
    std::array<std::size_t, 2> written_by_sensor{{0, 0}};
    for (const std::size_t ray_index : selected_indices)
    {
        const WorldRay &ray = rays[ray_index];
        const std::array<float, 6> values{{
            static_cast<float>(ray.origin.x()),
            static_cast<float>(ray.origin.y()),
            static_cast<float>(ray.origin.z()),
            static_cast<float>(ray.hit.x()),
            static_cast<float>(ray.hit.y()),
            static_cast<float>(ray.hit.z())}};
        stream.write(
            reinterpret_cast<const char *>(values.data()),
            static_cast<std::streamsize>(sizeof(float) * values.size()));
        ++written_by_sensor[ray.sensor];
    }
    stream.close();
    if (!stream)
    {
        error = "failed to finish source-ray shard";
        return false;
    }
    const std::uintmax_t expected_size =
        24U + static_cast<std::uintmax_t>(selected_indices.size()) * 24U;
    std::error_code size_error;
    const std::uintmax_t actual_size =
        std::filesystem::file_size(output_path, size_error);
    if (size_error || actual_size != expected_size)
    {
        error = "source-ray shard size validation failed";
        return false;
    }

    manifest_stream_
        << frame.pose_index << ','
        << frame.frame_reference_timestamp_ns << ','
        << frame.frame_end_timestamp_ns << ','
        << frame.fused_point_count << ','
        << frame.eligible_by_sensor[0] << ','
        << frame.eligible_by_sensor[1] << ','
        << frame.eligible_count << ','
        << written_by_sensor[0] << ','
        << written_by_sensor[1] << ','
        << selected_indices.size() << ','
        << std::setprecision(17) << maximum_ray_length_error_m << ','
        << p99_ray_length_error_m << ','
        << output_path.string() << '\n';
    manifest_stream_.flush();
    if (!manifest_stream_)
    {
        error = "failed to append source-ray export manifest";
        return false;
    }
    const Eigen::Vector3d normalized_up = gravity_up_base.normalized();
    local_cloud_manifest_stream_
        << frame.pose_index << ','
        << frame.pose_timestamp_ns << ','
        << frame.message_header_timestamp_ns << ','
        << frame.frame_reference_timestamp_ns << ','
        << frame.frame_end_timestamp_ns << ','
        << frame.frame_end_timestamp_ns -
               frame.frame_reference_timestamp_ns
        << ",base_link_at_frame_end,"
        << std::setprecision(17)
        << normalized_up.x() << ','
        << normalized_up.y() << ','
        << normalized_up.z() << ','
        << frame.eligible_by_sensor[0] << ','
        << frame.eligible_by_sensor[1] << ','
        << front_pcd_path << ','
        << back_pcd_path << '\n';
    local_cloud_manifest_stream_.flush();
    if (!local_cloud_manifest_stream_)
    {
        error = "failed to append frame-end local-cloud manifest";
        return false;
    }
    exported_pose_indices_.insert(frame.pose_index);
    ++exported_count_;
    total_eligible_count_ += frame.eligible_count;
    total_written_count_ += selected_indices.size();
    maximum_ray_length_error_m_ =
        std::max(maximum_ray_length_error_m_, maximum_ray_length_error_m);
    maximum_p99_ray_length_error_m_ =
        std::max(
            maximum_p99_ray_length_error_m_,
            p99_ray_length_error_m);
    return true;
}

bool Exporter::writeSummary(std::string &error) const
{
    const std::filesystem::path summary_path =
        std::filesystem::path(config_.output_dir) / "summary.txt";
    std::ofstream summary(summary_path, std::ios::trunc);
    if (!summary)
    {
        error = "failed to create source-ray summary";
        return false;
    }
    summary << "schema=exact_source_ray_export_v1\n"
            << "target_frame_count=" << frames_.size() << '\n'
            << "expected_frame_count=" << config_.expected_frame_count << '\n'
            << "active_frame_count=" << active_pose_indices_.size() << '\n'
            << "exported_frame_count=" << exported_count_ << '\n'
            << "eligible_ray_count=" << total_eligible_count_ << '\n'
            << "written_ray_count=" << total_written_count_ << '\n'
            << "maximum_ray_length_error_m=" << std::setprecision(17)
            << maximum_ray_length_error_m_ << '\n'
            << "maximum_p99_ray_length_error_m="
            << maximum_p99_ray_length_error_m_ << '\n'
            << "endpoint_voxel_m=" << std::setprecision(17)
            << config_.endpoint_voxel_m << '\n'
            << "sensor_pcd_saved="
            << (config_.save_sensor_pcd ? "true" : "false") << '\n'
            << "sensor_pcd_semantics=eligible_hits_in_base_link_at_frame_end_before_world_transform_and_endpoint_deduplication\n"
            << "gravity_semantics=physical_up_unit_vector_in_base_link_at_frame_end_independent_of_PGO_pose\n"
            << "pgo_pose_semantics=frame_end_pose_keyed_by_frame_begin_timestamp\n"
            << "world_pose_source=" << config_.pgo_tum << '\n'
            << "block_identity_source=" << config_.manifest_csv << '\n';
    return static_cast<bool>(summary);
}

bool Exporter::finalize(std::string &error)
{
    error.clear();
    if (!configured_ || !config_.enable || finalized_)
        return true;
    if (manifest_stream_.is_open())
        manifest_stream_.close();
    if (local_cloud_manifest_stream_.is_open())
        local_cloud_manifest_stream_.close();
    if (!manifest_stream_ || !local_cloud_manifest_stream_)
    {
        error = "failed to close one or more export manifests cleanly";
        std::string summary_error;
        writeSummary(summary_error);
        finalized_ = true;
        return false;
    }
    if (exported_count_ != active_pose_indices_.size())
    {
        error =
            "source-ray export count mismatch: expected active " +
            std::to_string(active_pose_indices_.size()) + ", exported " +
            std::to_string(exported_count_);
        writeSummary(error);
        finalized_ = true;
        return false;
    }
    for (const int target_index : active_pose_indices_)
    {
        const std::string stem = targetStem(target_index);
        const std::filesystem::path ray_path =
            std::filesystem::path(config_.output_dir) / "srrays" /
            (stem + ".srrays");
        if (!std::filesystem::is_regular_file(ray_path))
        {
            error =
                "missing finalized source-ray shard: " + ray_path.string();
            std::string summary_error;
            writeSummary(summary_error);
            finalized_ = true;
            return false;
        }
        if (config_.save_sensor_pcd)
        {
            for (const char *sensor : {"front", "back"})
            {
                const std::filesystem::path pcd_path =
                    std::filesystem::path(config_.output_dir) /
                    "frame_end_base" / sensor / (stem + ".pcd");
                if (!std::filesystem::is_regular_file(pcd_path))
                {
                    error =
                        "missing finalized sensor PCD: " +
                        pcd_path.string();
                    std::string summary_error;
                    writeSummary(summary_error);
                    finalized_ = true;
                    return false;
                }
            }
        }
    }
    if (!writeSummary(error))
    {
        finalized_ = true;
        return false;
    }
    finalized_ = true;
    return true;
}

}  // namespace fast_lio::source_ray_export
