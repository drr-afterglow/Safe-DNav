import pytest
import torch

from perception_training.geometry import (
    PanoramaSpec,
    absolute_to_relative_spherical_velocity,
    align_previous_body_frame,
    deskew_points_body,
    integrate_body_angular_velocity,
    make_network_input,
    quat_rotate,
    transform_world_to_body,
)
from perception_training.layers import SphericalConvBlock
from perception_training.isaac_labels import build_supervised_batch_from_isaac, validate_supervised_batch
from perception_training.losses import PerceptionLoss
from perception_training.model import SAUMNet
from perception_training.postprocess import build_rl_observation
from perception_training.runtime import LearnedPerceptionModule


def test_model_and_observation_shapes():
    spec = PanoramaSpec(height=16, width=180, max_range=10.0, vfov_deg=(-7.0, 52.0))
    curr = torch.randn(4, 128, 3)
    prev = curr + 0.1 * torch.randn_like(curr)
    curr[..., 0] += 4.0
    prev[..., 0] += 4.0
    x = make_network_input(curr, prev, spec)
    assert x.shape == (4, 4, 16, 180)
    assert set(torch.unique(x[:, 2:]).tolist()).issubset({0.0, 1.0})
    model = SAUMNet(height=16, width=180, vfov_deg=(-7.0, 52.0), base_channels=16)
    pred = model(x, torch.full((4,), 0.096))
    assert pred["mask_logits"].shape == (4, 1, 16, 180)
    assert pred["velocity"].shape == (4, 3, 16, 180)
    assert pred["log_sigma"].shape == (4, 3, 16, 180)
    static, dyn = build_rl_observation(
        x[:, 0:1],
        pred["mask_logits"],
        pred["velocity"],
        pred["log_sigma"],
        8.0,
        dynamic_channels=5,
        output_size=(6, 36),
    )
    assert static.shape == (4, 1, 6, 36)
    assert dyn.shape == (4, 5, 6, 36)

    batch = {
        "input": x,
        "mask": torch.zeros(4, 1, 16, 180),
        "velocity": torch.zeros(4, 3, 16, 180),
        "valid": torch.ones(4, 1, 16, 180),
    }
    loss = PerceptionLoss()(pred, batch)["loss"]
    assert torch.isfinite(loss)


def test_radial_velocity_weight_scales_gradient_without_changing_mean_weight():
    velocity_pred = torch.zeros(1, 3, 1, 1, requires_grad=True)
    pred = {
        "mask_logits": torch.zeros(1, 1, 1, 1),
        "velocity": velocity_pred,
        "log_sigma": torch.zeros(1, 3, 1, 1),
    }
    batch = {
        "input": torch.ones(1, 4, 1, 1),
        "mask": torch.ones(1, 1, 1, 1),
        "velocity": torch.ones(1, 3, 1, 1),
        "valid": torch.ones(1, 1, 1, 1),
    }
    criterion = PerceptionLoss(
        lambda_seg=0.0,
        lambda_static=0.0,
        velocity_component_weights=(2.0, 1.0, 1.0),
    )
    criterion(pred, batch)["loss"].backward()
    gradient = velocity_pred.grad.abs().flatten()
    torch.testing.assert_close(gradient[0], 2.0 * gradient[1])
    torch.testing.assert_close(gradient[1], gradient[2])
    torch.testing.assert_close(
        criterion.velocity_component_weights.mean(), torch.tensor(1.0)
    )


def test_far_positive_weight_only_increases_far_positive_focal_penalty():
    pred = {
        "mask_logits": torch.full((1, 1, 1, 2), -2.0),
        "velocity": torch.zeros(1, 3, 1, 2),
        "log_sigma": torch.zeros(1, 3, 1, 2),
    }
    base_batch = {
        "input": torch.tensor([[[[0.2, 0.8]], [[0.2, 0.8]], [[1.0, 1.0]], [[1.0, 1.0]]]]),
        "mask": torch.tensor([[[[1.0, 0.0]]]]),
        "velocity": torch.zeros(1, 3, 1, 2),
        "valid": torch.ones(1, 1, 1, 2),
    }
    weighted = PerceptionLoss(dice_weight=0.0, far_positive_weight=2.0)
    near_loss = weighted(pred, base_batch)["loss_seg"]
    far_batch = dict(base_batch)
    far_batch["mask"] = torch.tensor([[[[0.0, 1.0]]]])
    far_loss = weighted(pred, far_batch)["loss_seg"]
    assert far_loss > near_loss


def test_final_architecture_and_delta_t_scaling():
    model = SAUMNet(reference_delta_t=0.096).eval()
    assert isinstance(model.enc1.net[0], SphericalConvBlock)
    assert isinstance(model.enc1.net[1], SphericalConvBlock)
    assert isinstance(model.up2.conv.net[0], SphericalConvBlock)
    assert isinstance(model.up2.conv.net[1], SphericalConvBlock)
    assert tuple(model.pool2(model.pool1(torch.zeros(1, 16, 16, 180))).shape[-2:]) == (4, 45)

    x = torch.rand(1, 4, 16, 180)
    torch.nn.init.zeros_(model.uncertainty_head.weight)
    torch.nn.init.zeros_(model.uncertainty_head.bias)
    with torch.no_grad():
        at_reference = model(x, torch.tensor([0.096]))
        at_short_interval = model(x, torch.tensor([0.064]))
    torch.testing.assert_close(at_short_interval["velocity"], 1.5 * at_reference["velocity"])
    torch.testing.assert_close(
        torch.exp(at_short_interval["log_sigma"]),
        1.5 * torch.exp(at_reference["log_sigma"]),
    )


def test_absolute_velocity_is_converted_to_relative_flu_spherical_components():
    # A one-pixel panorama over a symmetric vertical FOV points along +x, so
    # its local bases are +x (radial), +y (azimuth), +z (pitch).
    absolute = torch.zeros(1, 3, 1, 1)
    ego_velocity_flu = torch.tensor([[1.0, 2.0, 3.0]])
    relative = absolute_to_relative_spherical_velocity(
        absolute,
        ego_velocity_flu,
        vfov_deg=(-1.0, 1.0),
    )
    torch.testing.assert_close(relative[0, :, 0, 0], torch.tensor([-1.0, -2.0, -3.0]))


def test_absolute_label_uses_positive_up_pitch_component():
    spec = PanoramaSpec(height=16, width=180, max_range=10.0, vfov_deg=(-7.0, 52.0))
    batch = build_supervised_batch_from_isaac(
        curr_static_hits_w=torch.tensor([[[5.0, 0.0, 0.0]]]),
        curr_dyn_hits_w=torch.tensor([[[2.0, 0.0, 0.0]]]),
        prev_hits_w=torch.tensor([[[2.0, 0.0, 0.0]]]),
        is_dyn_hit=torch.tensor([[True]]),
        dyn_hit_idx=torch.tensor([[0]]),
        all_dyn_vel_w=torch.tensor([[0.0, 0.0, 1.0]]),
        pos_w=torch.zeros(1, 3),
        quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        spec=spec,
        delta_t=0.096,
    )
    assert batch["velocity"][:, 2].max() == 1.0
    assert batch["velocity"][:, :2].abs().max() == 0.0


def test_deskew_and_frame_alignment_keep_static_geometry_consistent():
    raw_points = torch.tensor([[[10.0, 0.0, 0.0], [9.9, 0.0, 0.0]]])
    deskewed = deskew_points_body(
        raw_points,
        point_times=torch.tensor([[1_000_000_000.0, 1_000_000_000.1]], dtype=torch.float64),
        linear_vel_body=torch.tensor([[1.0, 0.0, 0.0]]),
        angular_vel_body=torch.zeros(1, 3),
        frame_time=torch.tensor([1_000_000_000.1], dtype=torch.float64),
    )
    torch.testing.assert_close(deskewed[0, :, 0], torch.tensor([9.9, 9.9]), atol=1e-5, rtol=1e-5)

    world_points = torch.tensor([[[4.0, 1.0, 2.0], [2.0, -3.0, 1.0]]])
    prev_pos = torch.zeros(1, 3)
    prev_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    curr_pos = torch.tensor([[1.0, 2.0, 0.5]])
    half_sqrt = 2.0**-0.5
    curr_quat = torch.tensor([[half_sqrt, 0.0, 0.0, half_sqrt]])
    prev_body = transform_world_to_body(world_points, prev_pos, prev_quat)
    expected_curr = transform_world_to_body(world_points, curr_pos, curr_quat)
    aligned = align_previous_body_frame(prev_body, prev_pos, prev_quat, curr_pos, curr_quat)
    torch.testing.assert_close(aligned, expected_curr)


def test_runtime_marks_first_frame_invalid_and_neutral():
    network_spec = PanoramaSpec(height=16, width=180, max_range=10.0, vfov_deg=(-7.0, 52.0))
    rl_spec = PanoramaSpec(height=6, width=36, max_range=10.0, vfov_deg=(-7.0, 52.0))
    runtime = LearnedPerceptionModule(
        checkpoint=None,
        spec=network_spec,
        rl_spec=rl_spec,
        dyn_max_vel=5.0,
        device="cpu",
    )
    points = torch.tensor([[[2.0, 0.0, 0.0], [3.0, 1.0, 0.5]]])
    pose = torch.zeros(1, 3)
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    point_times = torch.tensor([[1_000_000_000.0, 1_000_000_000.1]], dtype=torch.float64)
    static, dynamic, valid = runtime.process_body_points(
        points,
        pose,
        quat,
        curr_linear_vel_body=torch.tensor([[1.0, 0.0, 0.0]]),
        curr_point_times=point_times,
        point_linear_vel_body=torch.tensor([[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]),
        point_angular_vel_body=torch.zeros(1, 2, 3),
        curr_frame_time=torch.tensor([1_000_000_000.1], dtype=torch.float64),
    )
    assert not valid.any()
    torch.testing.assert_close(runtime.prev_points_body[0, :, 0], torch.tensor([1.9, 3.0]), atol=1e-5, rtol=1e-5)
    assert static.shape == (1, 1, 6, 36)
    torch.testing.assert_close(dynamic[:, 0], torch.ones_like(dynamic[:, 0]))
    torch.testing.assert_close(dynamic[:, 1:], torch.zeros_like(dynamic[:, 1:]))

    _, _, valid = runtime.process_body_points(points, pose, quat, delta_t=0.096)
    assert valid.all()


def test_body_rate_integration_and_supervised_quality_guard():
    identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    yaw_rate = torch.tensor([[0.0, 0.0, torch.pi / 2.0]])
    rotated_quat = integrate_body_angular_velocity(identity, yaw_rate, dt=1.0)
    rotated_x = quat_rotate(rotated_quat, torch.tensor([[1.0, 0.0, 0.0]]))
    torch.testing.assert_close(rotated_x, torch.tensor([[0.0, 1.0, 0.0]]), atol=1e-5, rtol=1e-5)

    batch = {
        "input": torch.cat([torch.ones(1, 2, 16, 180), torch.ones(1, 2, 16, 180)], dim=1),
        "mask": torch.zeros(1, 1, 16, 180),
        "velocity": torch.zeros(1, 3, 16, 180),
        "valid": torch.ones(1, 1, 16, 180),
        "delta_t": torch.tensor([0.096]),
    }
    validate_supervised_batch(batch, expected_size=(16, 180))
    batch["velocity"][0, 0, 0, 0] = 1.0
    with pytest.raises(ValueError, match="zero outside"):
        validate_supervised_batch(batch, expected_size=(16, 180))


def test_nearest_dynamic_pool_gathers_one_consistent_feature_vector():
    curr_range = torch.tensor([[[[0.2, 0.8]]]])
    mask_logits = torch.full((1, 1, 1, 2), 10.0)
    velocity = torch.tensor([[[[10.0, 2.0]], [[0.0, 0.0]], [[0.0, 0.0]]]])
    # The first pixel is nearer, so it must supply the complete feature vector
    # even though the second pixel has much larger uncertainty.
    log_sigma = torch.log(torch.tensor([[[[0.1, 1.0]], [[0.1, 1.0]], [[0.1, 1.0]]]]))
    _, dynamic = build_rl_observation(
        curr_range,
        mask_logits,
        velocity,
        log_sigma,
        dyn_max_vel=10.0,
        dynamic_channels=5,
        output_size=(1, 1),
    )
    expected = torch.tensor([0.2, 1.0, 0.0, 0.0, 0.01])
    torch.testing.assert_close(dynamic[0, :, 0, 0], expected)


def test_labels_follow_noisy_input_zbuffer_and_return_validity():
    spec = PanoramaSpec(height=1, width=1, max_range=10.0, vfov_deg=(-1.0, 1.0))
    static_hits = torch.tensor([[[10.0, 0.0, 0.0], [2.1, 0.0, 0.0]]])
    dynamic_hits = torch.tensor([[[2.0, 0.0, 0.0], [10.0, 0.0, 0.0]]])
    is_dynamic = torch.tensor([[True, False]])
    common = {
        "curr_static_hits_w": static_hits,
        "curr_dyn_hits_w": dynamic_hits,
        "prev_hits_w": torch.tensor([[[2.2, 0.0, 0.0], [1.9, 0.0, 0.0]]]),
        "is_dyn_hit": is_dynamic,
        "dyn_hit_idx": torch.zeros(1, 2, dtype=torch.long),
        "all_dyn_vel_w": torch.tensor([[1.0, 0.0, 0.0]]),
        "pos_w": torch.zeros(1, 3),
        "quat_w": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        "spec": spec,
        "delta_t": 0.096,
    }

    # Range noise makes the static ray win the input pixel although the clean
    # dynamic ray was nearer. Labels must follow the input Z-buffer winner.
    swapped = build_supervised_batch_from_isaac(
        **common,
        curr_input_hits_w=torch.tensor([[[2.2, 0.0, 0.0], [1.9, 0.0, 0.0]]]),
        curr_return_valid=torch.tensor([[True, True]]),
        prev_return_valid=torch.tensor([[True, True]]),
    )
    assert swapped["mask"].item() == 0.0
    assert swapped["velocity"].abs().max().item() == 0.0
    validate_supervised_batch(swapped, expected_size=(1, 1))

    # A dropped dynamic return must not leave privileged mask or velocity truth.
    dropped = build_supervised_batch_from_isaac(
        **common,
        curr_input_hits_w=torch.tensor([[[10.0, 0.0, 0.0], [10.0, 0.0, 0.0]]]),
        curr_return_valid=torch.tensor([[False, False]]),
        prev_return_valid=torch.tensor([[True, True]]),
    )
    assert dropped["valid"].item() == 0.0
    assert dropped["mask"].item() == 0.0
    assert dropped["velocity"].abs().max().item() == 0.0
    validate_supervised_batch(dropped, expected_size=(1, 1))
