#ifndef DNAV_CONTROLLER_SAFETY_CONTROLLER_H
#define DNAV_CONTROLLER_SAFETY_CONTROLLER_H

#include <dnav_controller/controller_types.h>

namespace dnav_controller {

class SafetyController {
 public:
  explicit SafetyController(const SafetyConfig& config);

  ControllerOutput update(const ControllerInput& input);
  void resetLimiter();
  const SafetyConfig& config() const { return config_; }

 private:
  SafetyConfig config_;
  Vector3 previous_velocity_world_;
  bool limiter_initialized_ = false;

  static bool finite(double value);
  static bool finite(const Vector3& value);
  static Vector3 worldToHeading(const Vector3& world, double yaw);
  ControllerOutput inactive(ControllerState state, const char* reason);
};

}  // namespace dnav_controller

#endif
