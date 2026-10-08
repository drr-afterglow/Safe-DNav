# NavRL 感知训练模块

这个目录实现了 PDF 方案中的点云到环境观测网络：输入当前帧和上一帧对齐后的高分辨全景距离图，预测动态 mask、障碍物绝对速度和不确定性；运行时再结合无人机速度构造相对速度，并通过风险保留降采样生成下游 RL 观测。

## 结构

- `perception_training/model.py`: 最终版 SA-UMNet，采用 `16→32→64→32→16` 对称 U-Net、`2x2` 最大池化、全卷积水平循环填充和最外层俯仰角调制 DCNv2。
- `perception_training/geometry.py`: IMU 去畸变、帧间坐标对齐、点云全景投影。
- `perception_training/mid360.py`: MID-360 非重复扫描配置；原始点云先编码成 `16x180` 感知输入，网络输出才降采样为 `6x36` RL 观测。
- `perception_training/losses.py`: Focal mask loss、异方差速度 loss、静态速度为零的自监督约束。
- `perception_training/isaac_labels.py`: 从 Isaac 仿真真值张量生成训练样本。
- `perception_training/runtime.py`: 部署适配器，将高分辨预测降采样为 `lidar_static` 和 `lidar_dynamic`。
- `docs/`: 感知表示契约、评估体系、训练策略和当前任务进度，入口见 `docs/README.md`。
- `data_collection/`: 随机位姿监督数据采集环境，静态/动态障碍物数量可配置。
- `train.py`: 离线训练脚本，已接入 wandb。
- `visualize_samples.py`: 生成四通道深度、valid、帧差、mask 和三维速度诊断图。

## MID-360 配置

配置文件默认按 MID-360 近似设置：

```text
水平 FOV: 360 deg
垂直 FOV: [-7, 52] deg
采样模式: non-repetitive full-FOV approximation
标称点频: 200000 points/s
扫描周期: 0.096 s
仿真计算子采样: 5760 points/frame
网络输入: 180 x 16 panorama
RL 接口: 36 x 6 panorama
```

`36x6` 只属于 PPO 接口，不参与感知数据采样。仿真中的5760点是对标称约19200点/帧的计算子采样，使用覆盖完整视场、跨帧不重复的低差异序列；真实部署时直接处理完整 MID-360 扫描，经相同去畸变和投影得到 `W=180,H=16` 输入。得到真实雷达包后，应依据有效回波率、全景占用率和距离分布校准仿真子采样数量、丢包率与噪声。

## 训练数据格式

每个 `.pt/.pth/.npz` 样本需要包含：

```text
input:    [2, H, W]  当前帧 Rt 和对齐后的上一帧 Rt-1，建议归一化到 [0, 1]
mask:     [1, H, W]  动态障碍物 mask
velocity: [3, H, W]  障碍物绝对速度在当前 FLU 机体系下的 (v_r, v_theta, v_phi)
delta_t: scalar  两帧实际时间间隔（秒）；旧数据缺失时使用配置中的参考间隔
valid:    [1, H, W]  可选，有效监督区域
```

## 采集随机位姿监督数据

采集代码在 `perception_training/data_collection/`。它会启动一个独立 Isaac 环境，生成静态地形障碍物和球形动态障碍物；每个样本随机放置无人机/雷达位姿，先采上一帧，再推进动态障碍物若干步采当前帧，最后保存：

```text
  input: 当前/上一帧合并后的高分辨原始雷达距离图
mask: 动态障碍物真值 mask
velocity: 动态障碍物绝对速度在当前机体系下的 (v_r, v_theta, v_phi)
delta_t: 两帧之间的实际仿真时间间隔
ego_velocity_body_mps: 当前 FLU 机体系三维自机线速度 [前、左、上]
valid: 有效监督区域
```

启动示例：

```bash
python perception_training/data_collection/collect.py \
  --config perception_training/data_collection/config/default.yaml \
  env.num_static_obstacles=200 \
  env_dyn.num_obstacles=80 \
  collection.num_samples=20000
```

常改配置：

```text
env.num_static_obstacles      静态障碍物数量
env_dyn.num_obstacles         动态障碍物数量
env_dyn.vel_range             动态障碍物速度范围
collection.pose_xy_range      随机采样的水平范围
collection.pose_z_range       随机采样的高度范围
collection.min_dynamic_pixels 至少包含多少动态像素才保存
```

默认输出到 `perception_training/data/mid360_random_pose/`，该目录可以直接写入 `config/default.yaml` 的 `data.roots` 后用于 `train.py`。

## 启动训练

```bash
python perception_training/train.py --config perception_training/config/default.yaml
```

wandb 配置在 `config/default.yaml` 的 `wandb` 字段中。默认 `mode: online`，如果只想本地调试可改成 `offline` 或把 `enabled` 设为 `false`。

## 训练策略建议

详细说明见 `docs/STRATEGY.md`。当前阶段只使用 `data_collection` 做固定/随机位姿离线采集，优先保证动态障碍物正样本、遮挡关系和速度标签准确。

## 接入 RL

当前优化方案的动态观测是 5 通道：

```text
[depth_dynamic, v_r, v_theta, v_phi, sigma]
```

网络监督的速度是障碍物绝对速度；部署时向 `process_body_points` 传入 `curr_linear_vel_body` 后，运行时会将其转换为这里使用的相对速度。完整定义见 `docs/REPRESENTATION.md`。

运行时返回 `(lidar_static, lidar_dynamic, history_valid)`。第一帧的动态观测为中性值且 `history_valid=false`；逐点去畸变所需的时间戳和 IMU 参数见 `docs/REPRESENTATION.md`。

`LearnedPerceptionModule(dynamic_channels=5)` 会默认输出完整格式。若需要兼容旧的 3 通道 PPO checkpoint，可以设置 `dynamic_channels=3`，输出为：

```text
[depth_dynamic, v_radial, v_tangent]
```

使用 3 通道还是 5 通道必须与已训练 PPO checkpoint 的动态卷积输入保持一致。

## 感知评估

训练与验证已接入完整 epoch 级指标，包括动态分割、距离分桶、三维速度、静态虚假速度、不确定性校准、Max-Danger Pooling 保留率和全景接缝表现。指标定义与解释见 `docs/EVALUATION.md`。最佳 checkpoint 使用分割与速度误差组成的综合评分，不再只依赖 IoU。

## 样本可视化

```bash
PYTHONPATH=/root/NavRL conda run -n NavRL python perception_training/visualize_samples.py \
  /path/to/scene_101 /path/to/scene_102 /path/to/scene_103 \
  --output-dir /path/to/visual_review --count 50 --seed 20260729
```

脚本为每个样本输出当前/上一帧深度、两张回波 valid、帧差、动态 mask、三轴速度、速度模长和自运动信息，并额外生成 `contact_sheet.png` 与可复现抽样记录 `selection.yaml`。
