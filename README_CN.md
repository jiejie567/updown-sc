# UpDown-SC：重力方向规范化的双包络 Scan Context

[English](./README.md) | 中文

[**项目主页**](https://jiejie567.github.io/updown-sc/) ·
[**可复现性与结果溯源**](docs/PROVENANCE.md) ·
[**AI 部署 Skill**](.agents/skills/updown-sc-deployment/SKILL.md)

本仓库公开预印本《UpDown-SC: A Gravity-Canonicalized Dual-Envelope Scan Context for Indoor LiDAR Place Recognition》的代码、基线、评估协议和复现材料。

论文中的每项数值结果都可追溯到数据包中的逐查询 CSV，详见 `docs/PROVENANCE.md`。

## 目录结构

| 目录 | 内容 |
|---|---|
| `updown_sc/` | 独立的 UpDown-SC C++ 实现，无 ROS 依赖：采用重力方向规范化和自适应分层的双包络 SCD 构建工具（`scan_context_rebuild`）、带掩码及偏航角假设的非均匀检索工具（`scan_context_cross_sequence_evaluator`），以及 SCD 转换和子集提取工具。 |
| `baselines/python/` | 已审查、与原方法公式等价的 CPU 基线实现（SC、SC++ (PC)、SOLiD、M2DP、RING++ CPU 移植版），以及查询与重力数据导出工具。 |
| `baselines/adapters/` | 官方 C++ 核心的适配器，包括 LiDAR-Iris 基准测试程序；BTC/STD 通过 `experiments/` 下的脚本运行。官方核心从各自的上游仓库获取，本仓库不重新分发。 |
| `experiments/` | 各数据集的实验协议脚本与 README，包括自采双会话、RTK-SLAM Construction Hall、Newer College、室内跨设备和 M2DGR hall 实验，以及使用官方检查点的 OverlapTransformer 和 MinkLoc3Dv2 适配器。 |
| `figures/` | 确定性的图表生成器，以及对应的源数据文件和完整性元数据。 |
| `docs/` | 结果溯源文档（论文数值到 CSV 的映射）和实验协议清单。 |
| `data/` | 精简结果包、校验和，以及完整评估数据包的清单，详见 `data/README.md`。 |

## 发布范围

本发布以**已去畸变并完成运动补偿的关键帧点云**为起点。数据包包含所有建图会话（关键帧 PCD、TUM 位姿、逐关键帧重力方向、描述子参数和 SCD 数据库）、所有查询集（去畸变的单帧 bin 文件、重力信息和元数据），以及所有已记录的逐查询结果 CSV。

生成这些会话的激光雷达惯性前端，即基于 FAST-LIO2 的 ROS 2 流水线，**不在本次发布范围内**；驱动该前端的实验脚本仅作为实验协议文档保留。论文中的全部检索结果均可仅使用已发布的点云复现。验证表明，从随附会话重新生成描述子，可得到逐字节一致的 SCD 数据库和相同的召回率。

## 构建 UpDown-SC

依赖：CMake >= 3.16、支持 C++17 的编译器、Eigen3、PCL（common + io）和 yaml-cpp。无需 ROS。

```bash
cmake -S updown_sc -B build/updown-sc -DCMAKE_BUILD_TYPE=Release
cmake --build build/updown-sc -j
```

工具用法（代码块中的注释与英文版保持一致）：

```bash
# Rebuild an SCD database from a session directory
# (key_point_frame/*.pcd + optimized_poses_tum.txt + scan_context_gravity.csv
#  + runtime_params.yaml):
build/updown-sc/scan_context_rebuild <session_dir> <out.scd> gravity <runtime_params.yaml> \
    <origin_height_m> [fixed_split_m]

# Cross-sequence retrieval (candidates CSV consumed by the evaluation scripts):
build/updown-sc/scan_context_cross_sequence_evaluator <map.scd> <query.scd> <out.csv> \
    <shortlist_K> <w_low> <w_high> <min_joint_rings> <dist_thresh> <margin>
# Paper setting: 100 0.3 0.7 2 0.5 0.1
```

标准描述子参数为 16 个环、60 个扇区、30 m 半径、0.25 m 体素，并启用双包络，配置位于 `updown_sc/config/descriptor_params.yaml`。每个已发布会话均附带构建时使用的 `runtime_params.yaml`。

## Python 环境

不依赖 ROS 的评估和绘图工具使用 Python 3.10+：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

重新生成图表或 PPT 所需的额外可选依赖列在 `requirements-figures.txt` 中。学习型方法的适配器使用 `docs/PROVENANCE.md` 记录的精确上游环境，其模型代码和权重不随本仓库分发。

## 复现论文表格

1. 使用 `data/package/` 中已校验的精简结果包，或下载 `data/README.md` 中描述的完整数据包，包括查询 bin 文件、重力 CSV、建图会话、SCD 数据库和所有已记录的逐查询 CSV。
2. 无需重新运行检索即可重算统计量：`experiments/common/compute_retrieval_stats.py` 可重新生成表 II 中不确定性说明及图 6 所用的 Wilson 区间和 F1max/AUPR；`compute_yaw_error.py` 可重新生成偏航角初始估计的数值结果。
3. 从输入重新运行检索：本方法使用上述 UpDown-SC 工具；Python 基线使用 `experiments/common/rescore_native_baselines.py` 和 `run_gravity_frontend_transfer.py`；官方 Iris 核心使用 `baselines/adapters/lidar_iris/benchmark.cpp`；两项学习型方法使用 `experiments/common/run_overlap_transformer.py` 和 `run_minkloc3dv2.py`。后两者分别使用上游 [OverlapTransformer](https://github.com/haomo-ai/OverlapTransformer) 和 [MinkLoc3Dv2](https://github.com/jac99/MinkLoc3Dv2) 仓库，以及官方在 KITTI 和 Oxford 上训练的检查点。本仓库不重新分发这些上游代码或权重。每个适配器在运行前校验检查点的 SHA-256，并在 `run_manifest.json` 中记录上游提交和完整实验协议。

归档中的路径字段使用可移植的 `${UPDOWN_SC_ROOT}` 占位符，路径约定详见 `data/PATHS.md`。

发布前检查会构建全部四个 C++ 工具，验证 Python 入口和归档校验和，并重放已发布的 M2DGR 检索：

```bash
./scripts/release_check.sh
```

## 许可证

项目自行编写的代码采用 MIT 许可证，见 `LICENSE`。官方基线核心和检查点（LiDAR-Iris、BTC、STD、OverlapTransformer、MinkLoc3Dv2）保留各自的上游许可证，通过外部获取而非在本仓库重新分发。数据集、基线和媒体的来源与署名记录在 `THIRD_PARTY_NOTICES.md` 和 `docs/assets/ATTRIBUTION.md` 中。

## 引用

请参阅 `CITATION.cff`。预印本发布后将补充 arXiv 标识符。
