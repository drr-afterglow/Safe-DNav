# SA-UMNet 感知表示契约

本文档冻结训练数据、网络输出和下游观测的物理语义。修改这些定义时必须同步更新数据采集、损失、运行时和 PPO/CBF 接口。

## 全景输入

- 每个网络帧是一段固定扫描周期内的 MID-360 点云；点云先利用逐点时间戳和 IMU 去畸变到扫描结束时刻，再投影为一张全景距离图。
- 网络输入形状冻结为 `[B, 4, 16, 180]`，通道顺序为 `[current_depth, aligned_previous_depth, current_return_valid, aligned_previous_return_valid]`。
- 距离除以 `max_range` 归一化；无回波像素填充为 `1.0`。两个有效性通道来自真实射线命中信息，取值为 `0/1`，不再只根据距离阈值猜测。
- 旧的 2 通道探针样本只允许在数据加载时临时派生有效性通道，不得作为正式采集格式。
- 图像第 0 行对应最大俯仰角，第 `H-1` 行对应最小俯仰角。
- `delta_t` 是两帧参考时刻的真实时间差，单位为秒。默认参考值为 `0.096s`，允许范围为 `[0.05s, 0.15s]`。
- 不使用多个历史网络帧进行地图式累积；单个扫描周期内部的点可以组成一帧，但必须统一去畸变到同一参考时刻。
- `curr_point_times` 与 `curr_frame_time` 使用同一秒制时钟。去畸变速度既可按一帧常量提供为 `[B,3]`，也可将高频 IMU 插值到每个点并提供为 `[B,N,3]`。
- 感知采集不使用 `isaac-training` 中为 PPO 降本设置的 `36x6` 固定射线。仿真使用完整 MID-360 视场内跨帧变化的非重复序列，真实雷达和仿真点云最终都通过同一 Z-buffer 投影编码为 `16x180`。

## 坐标系和速度

机体系采用 FLU 约定：

- `x`：机头前方；
- `y`：机体左侧；
- `z`：机体上方。

网络速度监督是真实障碍物的绝对平移速度旋转到当前机体系后的球坐标分量，不包含无人机平移速度：

```text
V_abs = [v_r, v_theta, v_phi]
```

- `v_r > 0`：沿视线远离无人机；
- `v_theta > 0`：方位角增大，即从 `+x` 朝 `+y`；
- `v_phi > 0`：俯仰角增大，即向上。

自运动补偿后的双帧残差与这个绝对速度定义一致。PPO/CBF 所需的相对速度在运行时构造：先将无人机机体系笛卡尔速度投影到每个像素的局部球坐标基，再计算 `V_rel = V_abs - V_ego`。

## 不确定性

网络输出 `log_sigma`，其中 `sigma = exp(log_sigma)` 表示三个速度分量的偶然不确定性标准差，单位为 `m/s`。训练使用与该标准差对应的拉普拉斯似然，拉普拉斯尺度为 `sigma / sqrt(2)`。

速度和 `sigma` 都按 `reference_delta_t / delta_t` 转换到物理速度单位。下游动态观测使用三个分量标准差的均值。

分割损失冻结为 valid-aware 的 `Focal(alpha=0.75, gamma=2.0) + Dice` 初始配置；这些系数允许通过正式小规模消融调整，但不得退回只使用会导致全静态坍塌的 `alpha=0.25` 配置。

## 网络和下游输出

网络原始输出：

```text
mask_logits: [B, 1, 16, 180]
velocity:    [B, 3, 16, 180]  # V_abs
log_sigma:   [B, 3, 16, 180]
```

提供无人机机体系速度后，运行时输出的五通道动态观测为：

```text
[dynamic_depth, v_r_rel, v_theta_rel, v_phi_rel, sigma_mean]
```

外部 ROS/PX4 数据若采用 FRD 或 NED 坐标系，必须先转换到本契约的 FLU 机体系。

## CBF 输入契约

CBF 不直接使用网络的绝对速度头，而是由运行时适配层提供三个物理量：

```text
distance_m:                     当前障碍物距离，单位 m
relative_radial_velocity_mps:  径向相对速度，远离为正、接近为负，单位 m/s
combined_uncertainty_mps:      感知与无人机速度估计合成标准差，单位 m/s
```

网络仍监督障碍物绝对速度，运行时按 `V_rel = V_abs - V_ego` 构造相对速度。若感知误差与无人机速度估计误差独立，合成不确定性为：

```text
sigma_rel = sqrt(sigma_perception^2 + sigma_ego^2)
```

稳健 CBF 代价使用径向速度下置置信界 `v_r_lower = v_r_rel - k_sigma * sigma_rel`，并计算：

```text
h = distance_m - d_safe
B_robust = v_r_lower + gamma * h
violation = relu(-B_robust)
```

这是训练阶段的 CBF-derived 连续安全代价，不是部署时逐动作求解的硬 CBF-QP 安全过滤器。当 CBF 仅基于球形安全距离时，无人机角速度引入的 `-omega x p` 项不改变径向距离导数；若以后使用方向相关安全集，必须显式加入该项。

## 运行时调用

`process_body_points` 要求提供扫描结束时刻的世界位置和姿态。启用逐点去畸变时还必须提供点时间戳、逐点/帧内线速度、逐点/帧内角速度和扫描结束时间：

```python
lidar_static, lidar_dynamic, history_valid = module.process_body_points(
    curr_points_body,
    curr_pos_w=position_at_scan_end,
    curr_quat_w=quaternion_at_scan_end,
    delta_t=measured_frame_interval,
    curr_linear_vel_body=linear_velocity_at_scan_end,
    curr_point_times=point_timestamps,
    point_linear_vel_body=interpolated_linear_velocity,
    point_angular_vel_body=interpolated_angular_velocity,
    curr_frame_time=scan_end_timestamp,
)
```

第一帧没有时间基线，返回的 `history_valid` 为假，动态观测被置为中性值。第二帧开始必须提供有效范围内的 `delta_t`。
