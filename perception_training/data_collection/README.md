# Perception Data Collection

This folder collects supervised samples for the perception network in an Isaac environment.

## What It Does

For each batch:

1. Randomize the drone/lidar pose in the map.
2. Sample a static control case or a short 6-DoF drone motion segment.
3. Capture a previous MID-360-like lidar frame.
4. Move/accelerate the drone and dynamic spheres for a randomized number of simulation steps.
5. Capture the current frame with a different non-repetitive scan pattern.
6. Apply configurable relative-pose alignment noise.
7. Validate tensor shapes, finite values, normalized ranges, time intervals and zero static velocity labels.
8. Use simulator truth to save `mask`, absolute obstacle velocity in the current FLU body frame, and the actual frame interval `delta_t`.

The static RayCaster and dynamic sphere raycast are kept separate so dynamic labels remain available while the network input still looks like a raw lidar observation. Dynamic obstacles are maintained as simulator-truth tensors for raycasting and labels; they do not need to be spawned as USD rigid objects during this offline collection stage.

The perception collector does not use the PPO `36x6` ray grid and does not model MID-360 as a fixed multi-line lidar. Every capture replaces Orbit's bootstrap ray buffer with a full-FOV, temporally varying low-discrepancy pattern. Consecutive frames and parallel environments use different scan phases, approximating MID-360's non-repetitive sampling before all returns are projected to the SA-UMNet `16x180` panorama.

The nominal profile is `200000 points/s * 0.096 s = 19200 points/frame`. The default Isaac run uses a compute-aware 5760-point subset while preserving the angular and temporal distribution. This value is deliberately configurable and must be calibrated from real MID-360 recordings. The `36x6` tensor is produced only after network inference by risk-aware postprocessing.

## Run

```bash
PYTHONPATH=/root/NavRL conda run -n NavRL python perception_training/data_collection/collect.py \
  --config perception_training/data_collection/config/default.yaml \
  env.num_static_obstacles=200 \
  env_dyn.num_obstacles=80 \
  collection.num_samples=20000
```

The script writes `.pt` shards and a `manifest.yaml` summary to `collection.output_dir`.

On the current AutoDL image, run collection through the existing `NavRL` Conda environment. Its activation config supplies the Isaac, NumPy, PyTorch and `typing_extensions` dependencies. Directly launching the embedded `python.sh` with `PYTHONNOUSERSITE=1` is not valid for this image because it hides required packages. Do not install or upgrade Isaac's base dependencies to work around this.

Mask and velocity labels use the same noisy current-ray geometry, return-valid mask and panorama Z-buffer winner as the four-channel network input. A dropped return therefore cannot leave privileged dynamic truth in the saved label.

## Important Configs

```text
env.num_static_obstacles       static terrain obstacle count
env_dyn.num_obstacles          dynamic obstacle count
env_dyn.vel_range              dynamic obstacle speed range
env_dyn.max_accel              maximum sphere acceleration; avoids velocity jumps
sensor.sampling.nominal_point_rate_hz
                               nominal hardware point rate
sensor.sampling.scan_duration  time span represented by one raw frame
sensor.sampling.simulated_points_per_frame
                               Isaac compute subset; default 5760
sensor.sampling.return_dropout_prob
                               simulated missing-return probability
sensor.sampling.range_noise_std_m
                               raw radial range noise in metres
sensor.network_panorama_width  projected network width, default 180
sensor.network_panorama_height projected network height, default 16
sensor.rl_panorama_width       downstream PPO width, default 36
sensor.rl_panorama_height      downstream PPO height, default 6
collection.steps_between_frames_range
                               randomized inclusive step range
collection.sensor_motion       drone velocity/rate/acceleration ranges
collection.alignment_noise     relative-pose error applied to previous-frame alignment
collection.scene_id            scene grouping key used for leak-free validation split
collection.min_dynamic_pixels  drop samples with too few dynamic pixels
collection.keep_empty_ratio    keep this fraction of non-dynamic samples as negatives
collection.force_exit_after_success
                               exit after all shards are flushed; avoids
                               Isaac 2023.1 headless shutdown segfaults
collection.close_sim_app       explicitly call SimulationApp.close()
```

`force_exit_after_success` is enabled by default because Isaac Sim 2023.1 can
segfault while unloading extensions after successful headless collection. The
writer flushes all shards before this exit path.

## Output Sample

```text
input:       [4, H, W]  current/previous depth and their return-valid masks
mask:        [1, H, W]
velocity:    [3, H, W]
delta_t: scalar seconds
valid:       [1, H, W]
scan_duration: scalar seconds
raw_point_count: scalar
valid_return_fraction: scalar
panorama_occupancy: scalar
row_occupancy: [H]
consecutive_frame_overlap: scalar
frame_interval_steps: scalar
ego_translation_m: scalar
ego_rotation_deg: scalar
ego_linear_speed_mps: scalar
ego_angular_speed_deg_s: scalar
alignment_position_error_m: scalar
alignment_rotation_error_deg: scalar
scene_id: scalar
```

Point clouds are projected to the high-resolution `180x16` network panorama before being saved.

`manifest.yaml` records the full sampling profile and aggregate valid-return fraction. Once real MID-360 bags are available, compare at least row-wise occupancy, total panorama occupancy, valid-return fraction, points per frame, range histogram and consecutive-frame overlap before finalizing the simulation profile.

Dynamic objects remain spheres in this stage, but their velocity now approaches a desired trajectory under `env_dyn.max_accel` instead of changing discontinuously.

For meaningful validation, collect at least two different procedural scenes and give each one a distinct `collection.scene_id` and output directory. `split_dataset()` keeps complete scene IDs on one side of the train/validation boundary. Example overrides:

```bash
PYTHONPATH=/root/NavRL conda run -n NavRL python perception_training/data_collection/collect.py \
  --config perception_training/data_collection/config/default.yaml \
  env.static_seed=20 collection.scene_id=20 \
  collection.output_dir=perception_training/data/scene_20
```
