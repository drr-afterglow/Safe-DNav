#ifndef DNAV_CONTROLLER_OBSTACLE_SAFETY_FILTER_H
#define DNAV_CONTROLLER_OBSTACLE_SAFETY_FILTER_H

#include <vector>

#include <dnav_controller/controller_types.h>

namespace dnav_controller {

struct ObstaclePoint {
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
  // dr/dt in the body-frame radial direction. Positive is separating.
  double relative_radial_velocity = 0.0;
  double radial_velocity_uncertainty = 0.0;
  bool dynamic = false;
  // Relative tangential velocity in the horizontal body frame. Positive is
  // toward the left-hand tangent of the radial direction.
  double relative_tangential_velocity = 0.0;
};

struct ObstacleFilterConfig {
  double vehicle_radius = 0.30;
  double safety_margin = 0.50;
  double dynamic_safety_margin = 0.90;
  double effective_deceleration = 0.25;
  double latency_budget = 0.15;
  double min_obstacle_z = -0.45;
  double max_obstacle_z = 0.25;
  double max_obstacle_range = 4.0;
  double max_horizontal_speed = 0.5;
  double escape_speed_threshold = 0.05;
  double dynamic_confidence_scale = 2.0;
  double dynamic_uncertainty_velocity_cap = 0.30;
  // ORCA parameters used for moving obstacles. The legacy braking envelope
  // remains the fallback for static returns and malformed dynamic estimates.
  bool use_orca_for_dynamic = true;
  double orca_time_horizon = 2.0;
  double orca_time_step = 0.05;
  double orca_safety_distance = 0.30;
  double dynamic_obstacle_radius = 0.45;
  // Lateral tie-break used when a close mover is directly in the route. It
  // prevents noisy tangential estimates from changing the chosen side every
  // control cycle while ORCA still enforces the actual admissible set.
  double dynamic_lateral_escape_speed = 0.35;
};

struct ObstacleFilterOutput {
  Vector3 safe_velocity_body;
  bool active = false;
  bool escape_assist = false;
  unsigned int active_constraints = 0;
  unsigned int dynamic_constraints = 0;
  double minimum_surface_clearance = 0.0;
  double minimum_allowed_closing_speed = 0.0;
};

class ObstacleSafetyFilter {
 public:
  explicit ObstacleSafetyFilter(const ObstacleFilterConfig& config);

  ObstacleFilterOutput filter(const Vector3& preferred_velocity_body,
                              const Vector3& actual_velocity_body,
                              const std::vector<ObstaclePoint>& obstacles,
                              const Vector3& goal_direction_body = {}) const;
  const ObstacleFilterConfig& config() const { return config_; }

 private:
  ObstacleFilterConfig config_;
  static bool finite(double value);
  static bool finite(const Vector3& value);
};

}  // namespace dnav_controller

#endif
