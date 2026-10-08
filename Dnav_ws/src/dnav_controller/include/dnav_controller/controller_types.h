#ifndef DNAV_CONTROLLER_CONTROLLER_TYPES_H
#define DNAV_CONTROLLER_CONTROLLER_TYPES_H

#include <cstdint>
#include <string>

namespace dnav_controller {

struct Vector3 {
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
};

enum class ControllerState : std::uint8_t {
  DISABLED = 0,
  WAITING_FOR_ODOMETRY = 1,
  WAITING_FOR_FLIGHT_STATE = 2,
  WAITING_FOR_POLICY = 3,
  ACTIVE = 4,
  ESTOP = 5,
  FAULT = 6,
};

struct SafetyConfig {
  double max_horizontal_speed = 0.5;
  double max_ascent_speed = 0.4;
  double max_descent_speed = 0.3;
  double max_horizontal_acceleration = 0.8;
  double max_vertical_acceleration = 0.5;
  double min_altitude = 0.5;
  double max_altitude = 4.0;
  double altitude_soft_margin = 0.5;
};

struct ControllerInput {
  bool navigation_enabled = false;
  bool estop_latched = false;
  bool odometry_valid = false;
  bool attitude_valid = false;
  bool flight_state_valid = false;
  bool flight_state_flying = false;
  bool policy_valid = false;
  Vector3 policy_velocity_world;
  double altitude = 0.0;
  double yaw = 0.0;
  double dt = 0.0;
};

struct ControllerOutput {
  ControllerState state = ControllerState::DISABLED;
  std::string reason = "disabled";
  Vector3 raw_velocity_world;
  Vector3 limited_velocity_world;
  Vector3 command_velocity_heading;
  bool horizontal_speed_limited = false;
  bool vertical_speed_limited = false;
  bool acceleration_limited = false;
  bool altitude_limited = false;
};

const char* stateName(ControllerState state);

}  // namespace dnav_controller

#endif
