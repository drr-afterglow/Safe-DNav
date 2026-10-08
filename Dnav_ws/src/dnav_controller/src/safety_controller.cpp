#include <dnav_controller/safety_controller.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace dnav_controller {

namespace {

double clamp(double value, double lower, double upper) {
  return std::max(lower, std::min(upper, value));
}

}  // namespace

const char* stateName(ControllerState state) {
  switch (state) {
    case ControllerState::DISABLED: return "DISABLED";
    case ControllerState::WAITING_FOR_ODOMETRY: return "WAITING_FOR_ODOMETRY";
    case ControllerState::WAITING_FOR_FLIGHT_STATE: return "WAITING_FOR_FLIGHT_STATE";
    case ControllerState::WAITING_FOR_POLICY: return "WAITING_FOR_POLICY";
    case ControllerState::ACTIVE: return "ACTIVE";
    case ControllerState::ESTOP: return "ESTOP";
    case ControllerState::FAULT: return "FAULT";
  }
  return "UNKNOWN";
}

SafetyController::SafetyController(const SafetyConfig& config) : config_(config) {
  if (!(config_.max_horizontal_speed > 0.0) ||
      !(config_.max_ascent_speed > 0.0) ||
      !(config_.max_descent_speed > 0.0) ||
      !(config_.max_horizontal_acceleration > 0.0) ||
      !(config_.max_vertical_acceleration > 0.0) ||
      !(config_.min_altitude < config_.max_altitude) ||
      !(config_.altitude_soft_margin > 0.0) ||
      config_.altitude_soft_margin >
          0.5 * (config_.max_altitude - config_.min_altitude)) {
    throw std::invalid_argument("invalid safety-controller configuration");
  }
}

bool SafetyController::finite(double value) { return std::isfinite(value); }

bool SafetyController::finite(const Vector3& value) {
  return finite(value.x) && finite(value.y) && finite(value.z);
}

Vector3 SafetyController::worldToHeading(const Vector3& world, double yaw) {
  const double cosine = std::cos(yaw);
  const double sine = std::sin(yaw);
  return Vector3{
      cosine * world.x + sine * world.y,
      -sine * world.x + cosine * world.y,
      world.z,
  };
}

void SafetyController::resetLimiter() {
  previous_velocity_world_ = Vector3{};
  limiter_initialized_ = false;
}

ControllerOutput SafetyController::inactive(ControllerState state, const char* reason) {
  resetLimiter();
  ControllerOutput output;
  output.state = state;
  output.reason = reason;
  return output;
}

ControllerOutput SafetyController::update(const ControllerInput& input) {
  if (input.estop_latched) {
    return inactive(ControllerState::ESTOP, "emergency stop latched");
  }
  if (!input.odometry_valid) {
    return inactive(ControllerState::WAITING_FOR_ODOMETRY, "odometry unavailable or stale");
  }
  if (!input.attitude_valid || !finite(input.altitude) || !finite(input.yaw)) {
    return inactive(ControllerState::FAULT, "odometry attitude is invalid");
  }
  if (!input.flight_state_valid) {
    return inactive(ControllerState::WAITING_FOR_FLIGHT_STATE, "flight state unavailable or stale");
  }
  if (!input.flight_state_flying) {
    return inactive(ControllerState::DISABLED, "vehicle is not in FLYING state");
  }
  if (!input.navigation_enabled) {
    return inactive(ControllerState::DISABLED, "navigation is disabled");
  }
  if (!input.policy_valid) {
    return inactive(ControllerState::WAITING_FOR_POLICY, "policy command unavailable or stale");
  }
  if (!finite(input.policy_velocity_world)) {
    return inactive(ControllerState::FAULT, "policy command contains NaN or Inf");
  }
  if (!finite(input.dt) || input.dt <= 0.0 || input.dt > 1.0) {
    return inactive(ControllerState::FAULT, "control period is invalid");
  }

  ControllerOutput output;
  output.state = ControllerState::ACTIVE;
  output.reason = "active";
  output.raw_velocity_world = input.policy_velocity_world;
  Vector3 limited = input.policy_velocity_world;

  const double horizontal_speed = std::hypot(limited.x, limited.y);
  if (horizontal_speed > config_.max_horizontal_speed) {
    const double scale = config_.max_horizontal_speed / horizontal_speed;
    limited.x *= scale;
    limited.y *= scale;
    output.horizontal_speed_limited = true;
  }

  const double clipped_z = clamp(
      limited.z, -config_.max_descent_speed, config_.max_ascent_speed);
  if (clipped_z != limited.z) output.vertical_speed_limited = true;
  limited.z = clipped_z;

  if (limited.z < 0.0) {
    const double scale = clamp(
        (input.altitude - config_.min_altitude) / config_.altitude_soft_margin,
        0.0, 1.0);
    const double lower = -config_.max_descent_speed * scale;
    if (limited.z < lower) {
      limited.z = lower;
      output.altitude_limited = true;
    }
  } else if (limited.z > 0.0) {
    const double scale = clamp(
        (config_.max_altitude - input.altitude) / config_.altitude_soft_margin,
        0.0, 1.0);
    const double upper = config_.max_ascent_speed * scale;
    if (limited.z > upper) {
      limited.z = upper;
      output.altitude_limited = true;
    }
  }

  const Vector3 previous = limiter_initialized_ ? previous_velocity_world_ : Vector3{};
  double delta_x = limited.x - previous.x;
  double delta_y = limited.y - previous.y;
  const double horizontal_delta = std::hypot(delta_x, delta_y);
  const double max_horizontal_delta = config_.max_horizontal_acceleration * input.dt;
  if (horizontal_delta > max_horizontal_delta) {
    const double scale = max_horizontal_delta / horizontal_delta;
    delta_x *= scale;
    delta_y *= scale;
    limited.x = previous.x + delta_x;
    limited.y = previous.y + delta_y;
    output.acceleration_limited = true;
  }
  const double max_vertical_delta = config_.max_vertical_acceleration * input.dt;
  const double vertical_delta = clamp(
      limited.z - previous.z, -max_vertical_delta, max_vertical_delta);
  if (vertical_delta != limited.z - previous.z) output.acceleration_limited = true;
  limited.z = previous.z + vertical_delta;

  previous_velocity_world_ = limited;
  limiter_initialized_ = true;
  output.limited_velocity_world = limited;
  output.command_velocity_heading = worldToHeading(limited, input.yaw);
  if (!finite(output.command_velocity_heading)) {
    return inactive(ControllerState::FAULT, "converted command is invalid");
  }
  return output;
}

}  // namespace dnav_controller
