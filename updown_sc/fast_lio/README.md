# FAST-LIO 使用说明

这是车端使用的 ROS 2 FAST-LIO 包，当前主要支持 MID360/AIRY 格式的 `sensor_msgs/msg/PointCloud2` 点云，并拆成两个入口：

- `mapping.launch.py`：建图，运行 FAST-LIO 并保存 PCD。
- `loc.launch.py`：定位，加载已经建好的 PCD 地图和 Scan Context 数据库，不再新增地图。

## 编译

在工作空间根目录编译：

```bash
cd /path/to/slam
colcon build --packages-select fast_lio --symlink-install
source install/setup.zsh
```

确认当前使用的是这个工作空间里的包：

```bash
ros2 pkg prefix fast_lio
```

期望输出类似：

```bash
/path/to/slam/install/fast_lio
```

## 配置

主要配置文件：

```bash
fast_lio/config/mid360.yaml
```

常用项：

```yaml
common:
  lid_topic: "/driver/lidar/point_cloud/Data"
  imu_topic: "/driver/lidar/lidar_front/imu/Data"

map_file_path: "prior_map/scans.pcd"
preprocess:
  blind: 0.3
  blind_filter_shape: "cylinder" # sphere/cylinder; cylinder uses XY radius
  blind_z_min: -0.5
  blind_z_max: 2.0
  max_range: 30.0      # discard points farther than 30 m before mapping/localization
  tag_filter_mode: "strict" # off/other/low_confidence/strict
prior_map:
  voxel_leaf: 0.5       # ICP coarse stage
  voxel_leaf_fine: 0.25 # ICP refine stage for top candidates
  icp_refine_top_k: 5
  scan_context:
    enable: true
    database_path: ""  # 建图默认写 PCD/scans.scd；定位默认由 map_file_path 派生
    num_rings: 8
    num_sectors: 60
    max_radius: 20.0
    min_joint_rings: 2 # 每个 sector 至少需要两个双方共同有效的 ring
    sector_support_exponent: 0.5 # 全局 sector 支撑使用平方根惩罚
    candidate_top_k: 100
    yaw_top_k: 3       # SC sector-key 粗对齐后在 ±3 sector 精评，保留 3 个 yaw 进入 ICP
    distance_thresh: 0.5
    seed_xy_offset: 0.0
localization_health:
  auto_relocalize_enable: true
  unhealthy_consecutive_frames: 10
  min_effective_points: 100
  restart_on_timestamp_rollback: true
```

如果要使用前雷达单独点云，把 `lid_topic` 改成：

```yaml
lid_topic: "/driver/lidar/lidar_front/point_cloud/Data"
```

`prior_map.use_prior_map` 不需要手动切换：

- `mapping.launch.py` 会强制设为 `false`
- `loc.launch.py` 会强制设为 `true`

定位模式下，`localization_health` 会在 prior-map 重定位成功后生效：连续 `unhealthy_consecutive_frames` 帧出现有效点过少（`effct_feat_num < min_effective_points`），或检测到时间戳严重回退时，会清掉当前定位状态并重新进入 prior-map 重定位。frame gap、速度和 residual 只保留为诊断日志，不触发自动重定位。

健康诊断发现当前帧异常时，`/Odometry`、`/localization_pose` 和 `/path` 会跳过该帧；触发重新重定位后会清空 path 历史，并停止发布定位输出，直到 prior-map 重定位再次成功，避免 planner 使用不可信定位。

## 建图

启动 FAST-LIO 建图：

```bash
cd /path/to/slam
source install/setup.zsh
ros2 launch fast_lio mapping.launch.py
```

建图 launch 默认会同时录制 FAST-LIO 使用的输入话题，也就是配置里的 `common.lid_topic` 和 `common.imu_topic`。默认输出目录：

```bash
fast_lio/mapping_rosbag/
```

录包会使用 `fast_lio/config/rosbag_qos.yaml`，当前默认按驱动话题的 `best_effort` QoS 订阅。终端应能看到类似日志：

```text
Subscribed to topic '/driver/lidar/lidar_front/imu/Data'
Subscribed to topic '/driver/lidar/point_cloud/Data'
All requested topics are subscribed.
```

如果录出来的 bag 是空的，先确认上游驱动或 `ros2 bag play` 正在真正发布消息：

```bash
ros2 topic hz /driver/lidar/point_cloud/Data
ros2 topic hz /driver/lidar/lidar_front/imu/Data
```

只看到 publisher 不代表有数据在流动；如果 `topic hz` 没有频率输出，录包也会是空包。

如果不想录包：

```bash
ros2 launch fast_lio mapping.launch.py record_bag:=false
```

如果要指定录包目录：

```bash
ros2 launch fast_lio mapping.launch.py bag_output_dir:=/abs/path/to/mapping_rosbag
```

播放 rosbag：

```bash
ros2 bag play /path/to/bag
```

如果需要 QoS 覆盖，普通回放可以使用默认的 best-effort 配置：

```bash
ros2 bag play /path/to/bag \
  --qos-profile-overrides-path fast_lio/config/rosbag_qos.yaml
```

离线回放融合后的点云 `/driver/lidar/point_cloud/Data` 时，每帧点云较大，建议使用 reliable 配置，避免播放器到 FAST-LIO 的链路丢帧：

```bash
ros2 bag play /path/to/bag \
  --topics /driver/lidar/point_cloud/Data /driver/lidar/lidar_front/imu/Data \
  --qos-profile-overrides-path fast_lio/config/rosbag_qos_reliable.yaml \
  --read-ahead-queue-size 10000
```

对应 FAST-LIO 也要用 reliable 订阅，例如临时覆盖参数：

```bash
ros2 run fast_lio fastlio_mapping --ros-args \
  --params-file install/fast_lio/share/fast_lio/config/mid360.yaml \
  -p prior_map.use_prior_map:=true \
  -p pcd_save.pcd_save_en:=false \
  -p publish_tf:=false \
  -p common.lidar_qos_reliability:=reliable \
  -p common.imu_qos_reliability:=reliable \
  -p common.lidar_subscribe_qos_depth:=500 \
  -p common.imu_subscribe_qos_depth:=4000
```

也可以用一键回放建图 launch，同时启动 FAST-LIO、可选 RViz 和 reliable rosbag 播放：

```bash
ros2 launch fast_lio replay_mapping.launch.py \
  bag_path:=${UPDOWN_SC_ROOT}/rosbag/nav_debug_bag_2
```

常用参数：

```bash
ros2 launch fast_lio replay_mapping.launch.py \
  bag_path:=${UPDOWN_SC_ROOT}/rosbag/nav_debug_bag_2 \
  start_offset:=0 \
  rate:=3.0 \
  clock_hz:=20
```

`replay_mapping.launch.py` 默认以 `rate:=3.0` 回放、使用 `use_sim_time:=true`，并让 `ros2 bag play` 发布 `/clock`；需要实时速度时可显式传 `rate:=1.0`。如果不用 launch 内置播包而接实时话题，可手动传 `use_sim_time:=false play:=false`。

建图路线建议：

- 为了提高 Scan Context 初始重定位成功率，建图时尽量覆盖之后可能启动定位的位置。
- 不要只沿一条直线快速经过；在关键区域可以多走一点、从不同朝向经过，让 `.scd` 里有足够的 keyframe。
- 室内重复走廊或相似房间较多时，建议在门口、拐角、开阔处多覆盖一些视角，这些位置对全局候选区分更有帮助。
- 建图和定位最好使用同一套雷达外参、话题、点云坐标系和滤波配置。

建图结束时 `Ctrl+C`，会自动保存一份 PCD 和一份 Scan Context 数据库：

```bash
fast_lio/PCD/scans.pcd
fast_lio/PCD/scans.scd
```

建图完成后，把地图复制到定位默认读取的位置，并保留同名 `.scd`：

```bash
cp fast_lio/PCD/scans.pcd fast_lio/prior_map/scans.pcd
cp fast_lio/PCD/scans.scd fast_lio/prior_map/scans.scd
```

默认定位读取的地图位置是：

```bash
fast_lio/prior_map/scans.pcd
fast_lio/prior_map/scans.scd
```

## 定位

确保已有地图和 Scan Context 数据库：

```bash
fast_lio/prior_map/scans.pcd
fast_lio/prior_map/scans.scd
```

启动定位：

```bash
cd /path/to/slam
source install/setup.zsh
ros2 launch fast_lio loc.launch.py
```

如果要临时指定另一份 PCD 地图，可以传 `map_pcd`；Scan Context 数据库会默认使用同名 `.scd`：

```bash
ros2 launch fast_lio loc.launch.py map_pcd:=/abs/path/to/scans.pcd
```

再播放同类型数据：

```bash
ros2 bag play /path/to/bag
```

如果定位输入是融合后的 `/driver/lidar/point_cloud/Data`，建议按上面的 reliable QoS 方式播放，并让 FAST-LIO 使用 reliable 订阅；否则大点云离线回放时可能出现 `LiDAR frame gap too large`。

也可以用一键回放定位 launch，同时启动 FAST-LIO、RViz 和稳定的 reliable rosbag 播放：

```bash
ros2 launch fast_lio replay_loc.launch.py \
  bag_path:=${UPDOWN_SC_ROOT}/rosbag/nav_debug_bag_2
```

如果 bag 内话题和配置里的 `common.lid_topic` / `common.imu_topic` 不一致，可以指定 bag 内源话题，launch 会自动 remap 到配置话题。例如 `bag_01` 只有前雷达点云：

```bash
ros2 launch fast_lio replay_loc.launch.py \
  bag_path:=${UPDOWN_SC_ROOT}/rosbag/bag_01 \
  bag_lid_topic:=/driver/lidar/lidar_front/point_cloud/Data
```

常用参数：

```bash
ros2 launch fast_lio replay_loc.launch.py \
  bag_path:=${UPDOWN_SC_ROOT}/rosbag/nav_debug_bag_2 \
  start_offset:=35 \
  rate:=3.0 \
  clock_hz:=20
```

`replay_loc.launch.py` 默认以 `rate:=3.0` 回放、使用 `use_sim_time:=true`，并让 `ros2 bag play` 发布 `/clock`；需要实时速度时可显式传 `rate:=1.0`。如果不用 launch 内置播包而接实时话题，可手动传 `use_sim_time:=false play:=false`。

定位模式下：

- 会发布 `/prior_map` 供 RViz 显示先验地图
- 会发布 `/cloud_registered`
- 会发布 `/Odometry`
- 会发布 `/path`
- 会发布 `/scan_context_icp_candidates`，显示进入 ICP 的前 3 个 Scan Context 候选 keyframe
- 不会新增地图
- 不会自动保存 PCD
- 初始重定位每次只积累当前窗口；ICP 失败后清空当前窗口，并用后续最新点云重新积累再试
- 重定位开始后会同时缓存同步好的 LiDAR/IMU 帧；ICP 成功后恢复到重定位开始前的 FAST-LIO 状态，优先补跑这段缓存数据，再继续处理实时输入

## 离线重定位窗口测试

从 rosbag 中每 1 秒取一帧 FAST-LIO 去畸变 body 点云，逐段使用当前 Scan Context + ICP 重定位方法测试，并把每个窗口的最佳结果保存为 PCD。开启重力规范化后，bag 必须同时录制 FAST-LIO 直接从 ESKF 重力状态生成的 `/scan_context_gravity_up`：

```bash
cd ${UPDOWN_SC_ROOT}/slam
source install/setup.zsh

ros2 run fast_lio offline_relocalization_exporter \
  --bag /path/to/undistorted_body_bag \
  --config fast_lio/config/mid360.yaml \
  --bag-topic /cloud_registered_body \
  --input-is-undistorted \
  --gravity-topic /scan_context_gravity_up \
  --output-dir fast_lio/Log/relocalization_windows \
  --stride 1.0 \
  --sample 0.1
```

`--stride` 表示每隔多少秒取一个候选窗口。离线工具默认每个窗口只使用第一帧 FAST-LIO 去畸变点云，与在线重定位一致。重力规范化开启时只允许该单帧模式，并按相同 header 时间戳读取 ESKF 的物理向上方向；不会从经过先验地图对齐的 `/Odometry` 反推重力，也不会用原始加速度计平均值代替。

生成离线输入 bag 时同时记录两个同步话题：

```bash
ros2 bag record \
  /cloud_registered_body \
  /scan_context_gravity_up \
  -o /path/to/undistorted_body_bag
```

若已有独立的物理重力 CSV，也可使用 `--gravity-csv`；列必须为 `stamp,up_x,up_y,up_z`，且必须与 `/cloud_registered_body` 来自同一次 replay。

建图的手工回环 session 会同时生成 `scan_context_gravity.csv`，逐关键帧保存 ESKF
物理向上方向。PGO 后导出的 PCD、轨迹和 SCD 位姿完整采用优化器给出的 SE(3)，包括
roll/pitch，不再用物理重力覆盖姿态，也不再拟合地面或平移 Z。物理重力 sidecar 只用于
将关键帧局部点云规范化后生成 Scan Context，并计算描述子坐标系相对完整 PGO 位姿的
canonical yaw，从而与查询端重力规范化保持一致。旧 session 没有这份 sidecar 时仍无法
重建重力规范化的 SCD，需要重新 replay 建图后再导出。

若输入 bag 是 FAST-LIO 发布的去畸变 body-frame 点云（默认话题 `/cloud_registered_body`），传入
`--input-is-undistorted` 可跳过原始雷达预处理，避免对已经去畸变的点云再次按原始包解析。建图时
Scan Context 关键帧和在线先验地图重定位也直接使用 `feats_undistort`，之后再分别按各自的 leaf size 降采样。
可用 `publish.scan_bodyframe_stride_s` 和 `publish.scan_bodyframe_sample_s` 只发布抽样窗口，减少中间 bag 体积。
排查个别窗口时可重复传入 `--window INDEX`，只重定位并保存指定窗口的 registered/overlay PCD。
`--one-frame-per-window` 可显式指定默认的单帧行为；需要复现旧的多帧窗口测试时，传入 `--accumulate-window`。

双包络使用统一的离地物理切分高度 `dual_z_split_height`。当描述子点云原点不在地面时，
必须配置该原点的离地高度 `origin_height_from_ground`。每个点先按
`z_ground = z + origin_height_from_ground` 转成离地高度，再直接与统一的
`dual_z_split_height` 比较；SCD 也保存这个离地高度。这里必须填写“生成描述子的
base_link 原点离地高度”，不能把雷达安装高度和雷达到 base_link 的外参 Z 重复相加。
建图启用 `dual_z_split_auto` 时，`dual_z_split_height` 不参与分界选择：系统对整张地图的关键帧构造
cell-balanced 高度直方图（同一关键帧的每个 ring-sector 在每个高度 bin 最多投一票），
在 `dual_z_split_auto_min` 到 `dual_z_split_auto_max` 之间用约束 Otsu 准则选一次物理
分界，并用它统一重建全部地图描述子。上下层任一侧少于
`dual_z_split_auto_min_layer_fraction`，或关键帧少于
`dual_z_split_auto_min_keyframes` 时自动关闭双层并退化为标准单层 SC，而不是回退到人工
固定高度。最终模式与分界写入 SCD；定位加载地图后自动复用，不会在查询序列上再次估计。
`retrieval_height_offset` 默认 `0.1 m`，仅在构造检索 key 和计算余弦距离时临时加入；
不会修改 SCD 高度或 `Delta-z`。V7 SCD 会同时记录物理切分高度与建图平台原点高度；
修改建图平台高度后需要重新生成 `scans.scd`。

双包络计分会区分“没有证据”和“证据冲突”：某一包络在地图与查询两侧都没有达到
任何有效单元时，系统检查该地图候选在
`absent_upper_fallback_radius`（默认 10 m）邻域内的关键帧；邻域至少包含
`absent_upper_fallback_min_keyframes` 帧且上层观测比例不超过
`absent_upper_fallback_max_local_fraction`（默认 5%）时，才省略上包络并归一化剩余
权重。这使半室内半室外地图中的室外候选可退化为下包络匹配，而室内候选仍要求双包络。
任一侧存在稀疏上层单元但不足 `min_joint_rings` 时，整个候选按最大距离 1.0 惩罚，
避免把稀疏或局部缺失误当成好匹配。

启用 `vertical_estimation_enable` 后，查询端先按原始 SC 的方式用 sector key 在全部
sector 上粗对齐，再仅对粗对齐附近 ±3 sector 计算完整双包络距离，并为每个地点候选
保留 top-3 yaw；随后固定候选和 sector 位移，从共同有效的双包络单元直接估计连续
`Delta-z`。下包络保留最大值最低的 50%，上包络保留最小值最高的 50%，再对
保留残差计算通道加权中位数；不会根据 z 重建 mask。保留比例由
`vertical_stable_fraction` 控制，默认 0.5。最终以 `candidate_z + Delta-z` 初始化 ICP。
高度不会参与候选或 yaw 排名，结果限制在
`[vertical_correction_min, vertical_correction_max]` 内。该逻辑不改变已有 SCD
文件格式，无需重新生成先验地图。

当同一 ring-sector 的下包络最大值和上包络最小值都落在
离地切分线 `dual_z_split_height` 的
`± vertical_boundary_margin` 附近时，该格的层归属不稳定，因此只在
`Delta-z` 估计中跳过；主 SCD 候选和 yaw 得分仍使用原始双包络，不改变检索排名。该 mask
由现有 SCD 值和有效位运行时推导，不写入文件。当前
`vertical_boundary_margin=0.1 m` 约为 `voxel_leaf=0.25 m` 的 0.4 倍。

如果 bag 内点云话题和配置里的 `common.lid_topic` 不一致，用 `--bag-topic` 指定实际源话题。例如：

```bash
ros2 run fast_lio offline_relocalization_exporter \
  --bag ${UPDOWN_SC_ROOT}/rosbag/bag_01 \
  --config fast_lio/config/mid360.yaml \
  --bag-topic /driver/lidar/lidar_front/point_cloud/Data \
  --output-dir fast_lio/Log/relocalization_windows_bag_01 \
  --stride 1.0 \
  --sample 0.2
```

输出文件：

```text
fast_lio/Log/relocalization_windows/registered_windows/
fast_lio/Log/relocalization_windows/overlay_windows/
fast_lio/Log/relocalization_windows/relocalization_best_registered.pcd
fast_lio/Log/relocalization_windows/relocalization_best_overlay.pcd
fast_lio/Log/relocalization_windows/relocalization_windows.csv
fast_lio/Log/relocalization_windows/scan_context_candidate_hypotheses.csv
fast_lio/Log/relocalization_windows/scan_context_trajectory.csv
fast_lio/Log/relocalization_windows/relocalization_top_view.png
fast_lio/Log/relocalization_windows/relocalization_top_view_focus.png
```

其中 `registered_windows/` 保存每个窗口配准后的点云，`overlay_windows/` 保存灰色 prior map 加绿色窗口点云的叠加图。`relocalization_best_registered.pcd` 和 `relocalization_best_overlay.pcd` 是全局 fitness 最好的快捷副本，`relocalization_windows.csv` 记录每个窗口的 fitness、overlap、高度修正和失败原因。为兼容旧分析脚本，CSV 中的 `coarse_vertical_shift` 与 `vertical_shift` 当前记录同一个单次估计值；`scan_context_candidate_hypotheses.csv` 保存每个候选的 yaw、高度修正与最终 seed z。
离线工具默认还会从 `.scd` 导出建图关键帧轨迹，并生成完整地图俯视图和诊断区域放大图；图中灰色为 PCD、青色为 SCD 建图轨迹、绿色为评估通过位置、红色为评估拒绝位置。

如果有人工标注或连续里程计对齐得到的真实位置，可通过 `--truth-csv PATH` 叠加到两张 PNG。CSV 至少包含 `window,x,y`，也接受 `window,map_x,map_y`；真实位置显示为黄色星号，并连接到该窗口的重定位返回位置。没有真值 CSV 时只画评估结果，不推测真实位置。若运行环境不需要图片或没有 Matplotlib，可传入 `--no-summary-png`；绘图失败只会警告，不影响已经生成的 CSV/PCD。

## RViz

RViz 由 launch 自动打开，配置如下：

- 建图：`rviz/fastlio.rviz`
- 定位：`rviz/fastlio_prior.rviz`

如果不想打开 RViz：

```bash
ros2 launch fast_lio mapping.launch.py rviz:=false
ros2 launch fast_lio loc.launch.py rviz:=false
```

## 常用检查命令

查看话题：

```bash
ros2 topic list
```

确认点云频率：

```bash
ros2 topic hz /driver/lidar/point_cloud/Data
```

确认 IMU 频率：

```bash
ros2 topic hz /driver/lidar/lidar_front/imu/Data
```

查看点云字段：

```bash
ros2 topic echo --once /driver/lidar/point_cloud/Data --field fields
```

如果配置里改成了前雷达单独点云，则把上面的点云话题换成 `/driver/lidar/lidar_front/point_cloud/Data`。

当前适配的点云字段包括：

```text
x, y, z, intensity, tag, line, timestamp
```

`preprocess.tag_filter_mode` 控制 Livox `tag` 过滤：`off` 不过滤；`other` 沿用 FAST-LIO 原逻辑，只看 bit `[5:4]`；`low_confidence` 过滤三组里明确低置信的点，适合默认避障/定位；`strict` 只保留三组全高置信点。

`preprocess.blind_filter_shape: "cylinder"` 时，`blind` 表示 XY 半径，`blind_z_min/blind_z_max` 表示要过滤的 Z 范围；当前默认会过滤 `xy <= 0.3m` 且 `-0.5m <= z <= 2.0m` 的车体附近圆柱。

## 输出

主要输出话题：

```text
/Odometry
/path
/cloud_registered
/cloud_registered_body
/prior_map
/scan_context_icp_candidates
/localization_pose
```

默认坐标系：

```text
map -> base_link
```

## 注意

- 修改 `mid360.yaml` 后，如果使用的是 symlink install，一般不需要重新编译；改 C++ 或 launch 后建议重新编译。
- `loc.launch.py` 依赖 `map_file_path` 指向的 PCD 文件和同名 `.scd` 文件存在。
- 使用 `map_pcd:=...` 时也要保证同目录下有同名 `.scd`，或者在 `prior_map.scan_context.database_path` 指定数据库路径。
- 建图后记得把 `PCD/scans.pcd` 和 `PCD/scans.scd` 一起复制到 `prior_map/`。
- 新生成的 V7 `.scd` 直接保存离地高度、二进制 bitset 有效掩码、重力规范化标记、descriptor canonical yaw、离地物理切分高度和建图平台原点高度，不再使用旧 `height_offset`。加载 V6 时会利用其中的建图平台原点高度自动迁移为离地高度；V1--V4 会先减去旧高度偏置，V1--V5 因没有可靠的平台高度元数据，只能保持原数值兼容读取。也可由 `scan_context_convert` 转换成 V7；跨平台实验建议用正确的平台高度重新生成描述子。旧描述子仍保留原本的重力规范化标记，不能冒充新的重力规范化地图。
- 如果定位启动时报 Scan Context 数据库缺失，重新跑一次建图或回放并正常退出生成 `.scd`。
- 当前地图保存只保留退出自动保存：PCD 写到 `PCD/scans.pcd`，Scan Context 数据库写到 `PCD/scans.scd`。
- 回放 bag 内如果夹杂明显时间回退的 LiDAR/IMU 消息，节点会丢弃这些旧消息，避免破坏 FAST-LIO 的单调时间线。

## BTC 十帧子地图导出

BTC 的官方输入是由连续已配准扫描构成的子地图，不能直接使用当前重定位的
0.1 秒单帧。`replay_btc_submaps.launch.py` 会以纯 FAST-LIO 里程计模式回放原始
LiDAR/IMU bag，将当前帧及其前 9 帧去畸变点云根据 `/Odometry` 配准到当前帧，并输出紧凑
的 XYZI float32 `.bin` 子地图。该流程会关闭地图 PCD、SCD 和手工回环文件保存，
默认输出也位于 OneDrive 之外。

先编译：

```bash
cd ${UPDOWN_SC_ROOT}/OneDrive/icra2027/slam
source /opt/ros/jazzy/setup.zsh
colcon build --packages-select fast_lio --symlink-install
source install/setup.zsh
```

导出建图包：

```bash
ros2 launch fast_lio replay_btc_submaps.launch.py \
  bag_path:=${UPDOWN_SC_ROOT}/rosbag/mapping_2_floor \
  output_dir:=${UPDOWN_SC_ROOT}/icra2027_runtime/experiments/btc_submaps/mapping_2_floor \
  scans_per_submap:=10 submap_stride_scans:=10 voxel_size:=0.1 \
  rate:=2.0 overwrite:=true
```

导出定位包：

```bash
ros2 launch fast_lio replay_btc_submaps.launch.py \
  bag_path:=${UPDOWN_SC_ROOT}/rosbag/loc_2_floor \
  output_dir:=${UPDOWN_SC_ROOT}/icra2027_runtime/experiments/btc_submaps/loc_2_floor \
  scans_per_submap:=10 submap_stride_scans:=10 voxel_size:=0.1 \
  rate:=2.0 overwrite:=true
```

BTC 导出默认单独使用 `2.0` 倍速。启动文件会在 rosbag 到达 EOF 后等待
`drain_delay`（默认 5 秒），让 FAST-LIO 和可靠 QoS 队列排空；本机离线批处理可使用
`rate:=3.0 drain_delay:=15.0`，但必须检查 manifest 中的同步帧数和尾段时间。
用于检索评测时，建图 bag、优化轨迹和查询伪真值必须属于同一地图会话。例如当前
双包先导实验使用 `mapping_2_floor`、`manual_loop_session_gravity/optimized_poses_tum.txt`
和 `loc_2_floor`；`mapping_big` 是另一地图会话，不能直接沿用这组伪真值。

每个目录包含：

```text
submaps/000000.bin ...   # 当前帧 base_link 坐标系下的因果十帧配准子地图
metadata.csv             # 点数、时间、位姿和文件索引
poses_tum.txt            # 子地图当前帧的 FAST-LIO 局部里程计位姿
manifest.json            # 完整导出参数与丢帧统计
```
