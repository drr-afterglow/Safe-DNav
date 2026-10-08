#include <dnav_controller/obstacle_safety_filter.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace dnav_controller {

namespace {

struct HalfPlane {
  double nx;
  double ny;
  double limit;
};

struct Point2 {
  double x;
  double y;
};

double dot(const Point2& left, const Point2& right) {
  return left.x*right.x + left.y*right.y;
}

double normSquared(const Point2& value) { return dot(value, value); }

Point2 operator+(const Point2& left, const Point2& right) {
  return {left.x+right.x, left.y+right.y};
}

Point2 operator-(const Point2& left, const Point2& right) {
  return {left.x-right.x, left.y-right.y};
}

Point2 operator*(double scale, const Point2& value) {
  return {scale*value.x, scale*value.y};
}

Point2 clampToDisk(const Point2& value, double radius) {
  const double length = std::hypot(value.x, value.y);
  if (length <= radius || length <= 1e-12) return value;
  return {(radius/length)*value.x, (radius/length)*value.y};
}

double signedViolation(const HalfPlane& constraint, const Point2& point) {
  return constraint.nx*point.x + constraint.ny*point.y - constraint.limit;
}

bool satisfies(const HalfPlane& constraint, const Point2& point) {
  return signedViolation(constraint, point) <= 1e-9;
}

Point2 segmentIntersection(const Point2& from, const Point2& to,
                           double from_violation, double to_violation) {
  const double fraction = from_violation/(from_violation-to_violation);
  return {from.x+fraction*(to.x-from.x), from.y+fraction*(to.y-from.y)};
}

void clipPolygon(const HalfPlane& constraint, std::vector<Point2>* polygon) {
  if (polygon->empty()) return;
  constexpr double kTolerance = 1e-12;
  std::vector<Point2> clipped;
  clipped.reserve(polygon->size()+1);
  Point2 previous = polygon->back();
  double previous_violation = signedViolation(constraint, previous);
  bool previous_inside = previous_violation <= kTolerance;
  for (const Point2& current : *polygon) {
    const double current_violation = signedViolation(constraint, current);
    const bool current_inside = current_violation <= kTolerance;
    if (current_inside != previous_inside) {
      clipped.push_back(segmentIntersection(previous, current,
                                             previous_violation, current_violation));
    }
    if (current_inside) clipped.push_back(current);
    previous = current;
    previous_violation = current_violation;
    previous_inside = current_inside;
  }
  polygon->swap(clipped);
}

Point2 closestPointOnPolygon(const Point2& preferred,
                             const std::vector<Point2>& polygon) {
  Point2 closest{0.0, 0.0};  // Zero is feasible for every generated constraint.
  double best_squared_distance = std::numeric_limits<double>::infinity();
  for (std::size_t index=0; index<polygon.size(); ++index) {
    const Point2& start = polygon[index];
    const Point2& end = polygon[(index+1)%polygon.size()];
    const double dx = end.x-start.x;
    const double dy = end.y-start.y;
    const double length_squared = dx*dx+dy*dy;
    const double projection = length_squared > 0.0 ? std::max(
        0.0, std::min(1.0, ((preferred.x-start.x)*dx+
                            (preferred.y-start.y)*dy)/length_squared)) : 0.0;
    const Point2 candidate{start.x+projection*dx, start.y+projection*dy};
    const double error_x = preferred.x-candidate.x;
    const double error_y = preferred.y-candidate.y;
    const double squared_distance = error_x*error_x+error_y*error_y;
    if (squared_distance < best_squared_distance) {
      best_squared_distance = squared_distance;
      closest = candidate;
    }
  }
  return closest;
}

Point2 leastViolatingVelocity(const Point2& preferred,
                              const std::vector<HalfPlane>& constraints,
                              double speed) {
  Point2 best{0.0, 0.0};
  double best_max_violation = std::numeric_limits<double>::infinity();
  double best_squared_violation = std::numeric_limits<double>::infinity();
  double best_preference_error = std::numeric_limits<double>::infinity();
  constexpr int kDirections = 144;
  for (int index=-1; index<kDirections; ++index) {
    Point2 candidate{0.0, 0.0};
    if (index >= 0) {
      const double angle = 2.0*M_PI*index/kDirections;
      candidate = {speed*std::cos(angle), speed*std::sin(angle)};
    }
    double maximum_violation = 0.0;
    double squared_violation = 0.0;
    for (const auto& constraint : constraints) {
      const double violation = std::max(
          0.0, signedViolation(constraint, candidate));
      maximum_violation = std::max(maximum_violation, violation);
      squared_violation += violation*violation;
    }
    const double preference_error =
        std::pow(preferred.x-candidate.x, 2)+
        std::pow(preferred.y-candidate.y, 2);
    constexpr double kTolerance = 1e-12;
    if (maximum_violation < best_max_violation-kTolerance ||
        (std::abs(maximum_violation-best_max_violation) <= kTolerance &&
         (squared_violation < best_squared_violation-kTolerance ||
          (std::abs(squared_violation-best_squared_violation) <= kTolerance &&
           preference_error < best_preference_error)))) {
      best = candidate;
      best_max_violation = maximum_violation;
      best_squared_violation = squared_violation;
      best_preference_error = preference_error;
    }
  }
  return best;
}

bool satisfiesAll(const std::vector<HalfPlane>& constraints,
                  const Point2& candidate, double speed) {
  if (normSquared(candidate) > speed*speed + 1e-9) return false;
  for (const auto& constraint : constraints) {
    if (!satisfies(constraint, candidate)) return false;
  }
  return true;
}

// Find the Euclidean projection onto the intersection of half-planes and a
// circular speed envelope. This is the 2-D equivalent of the ORCA linear
// program used by the legacy ROS1 shield; unlike the old square polygon,
// every returned velocity satisfies the actual horizontal speed limit.
bool closestPointInDisk(const Point2& preferred,
                        const std::vector<HalfPlane>& constraints,
                        double speed, Point2* result) {
  std::vector<Point2> candidates;
  candidates.reserve(2*constraints.size()*constraints.size()+2);
  candidates.push_back(clampToDisk(preferred, speed));
  for (const auto& constraint : constraints) {
    const double length = std::hypot(constraint.nx, constraint.ny);
    if (length <= 1e-12) continue;
    const Point2 normal{constraint.nx/length, constraint.ny/length};
    const double offset = constraint.limit/length;
    const Point2 projection{preferred.x+(offset-dot(normal, preferred))*normal.x,
                            preferred.y+(offset-dot(normal, preferred))*normal.y};
    candidates.push_back(projection);
    const double distance = std::abs(offset);
    if (distance <= speed + 1e-9) {
      const double tangent = std::sqrt(std::max(0.0, speed*speed-distance*distance));
      const Point2 tangentDirection{-normal.y, normal.x};
      candidates.push_back(offset*normal + tangent*tangentDirection);
      candidates.push_back(offset*normal - tangent*tangentDirection);
    }
  }
  for (std::size_t first=0; first<constraints.size(); ++first) {
    for (std::size_t second=first+1; second<constraints.size(); ++second) {
      const auto& a = constraints[first];
      const auto& b = constraints[second];
      const double determinant = a.nx*b.ny-a.ny*b.nx;
      if (std::abs(determinant) <= 1e-12) continue;
      candidates.push_back({(a.limit*b.ny-a.ny*b.limit)/determinant,
                            (a.nx*b.limit-a.limit*b.nx)/determinant});
    }
  }
  bool found = false;
  double bestError = std::numeric_limits<double>::infinity();
  for (const auto& candidate : candidates) {
    if (!satisfiesAll(constraints, candidate, speed)) continue;
    const double error = normSquared(candidate-preferred);
    if (!found || error < bestError) {
      found = true;
      bestError = error;
      *result = candidate;
    }
  }
  return found;
}

bool maxProgressInDisk(const Point2& direction,
                       const std::vector<HalfPlane>& constraints,
                       double speed, Point2* result) {
  const double length = std::hypot(direction.x, direction.y);
  if (length <= 1e-12) return false;
  bool found = false;
  double best = -std::numeric_limits<double>::infinity();
  constexpr int kSamples = 720;
  for (int index=0; index<kSamples; ++index) {
    const double angle = 2.0*M_PI*index/kSamples;
    const Point2 candidate{speed*std::cos(angle), speed*std::sin(angle)};
    if (!satisfiesAll(constraints, candidate, speed)) continue;
    const double progress = dot(direction, candidate)/length;
    if (!found || progress > best) {
      found = true;
      best = progress;
      *result = candidate;
    }
  }
  Point2 closest;
  if (closestPointInDisk(speed*(1.0/length)*direction, constraints, speed, &closest)) {
    const double progress = dot(direction, closest)/length;
    if (!found || progress > best) {
      found = true;
      *result = closest;
    }
  }
  return found;
}

// Construct the ORCA plane for one moving return. The returned half-plane is
// n.v <= limit, matching the static braking constraints in this file.
bool makeOrcaConstraint(const ObstaclePoint& obstacle,
                        const Point2& agent_velocity,
                        const ObstacleFilterConfig& config,
                        const Point2& preferred,
                        HalfPlane* result) {
  const Point2 position{obstacle.x, obstacle.y};
  const double distance_squared = normSquared(position);
  if (distance_squared <= 1e-12) return false;
  const double distance = std::sqrt(distance_squared);
  const Point2 radial{position.x/distance, position.y/distance};
  const Point2 tangent{-radial.y, radial.x};
  const double uncertainty = std::min(
      config.dynamic_uncertainty_velocity_cap,
      config.dynamic_confidence_scale*std::max(0.0, obstacle.radial_velocity_uncertainty));
  const double relative_radial = obstacle.relative_radial_velocity-uncertainty;
  const Point2 relative_obstacle_velocity{
      relative_radial*radial.x + obstacle.relative_tangential_velocity*tangent.x,
      relative_radial*radial.y + obstacle.relative_tangential_velocity*tangent.y};
  const Point2 obstacle_velocity = agent_velocity + relative_obstacle_velocity;
  const Point2 relative_velocity = agent_velocity-obstacle_velocity;
  const double obstacle_radius = obstacle.dynamic
      ? std::max(config.orca_safety_distance, config.dynamic_obstacle_radius)
      : config.orca_safety_distance;
  const double combined_radius = config.vehicle_radius + obstacle_radius;
  const double radius_squared = combined_radius*combined_radius;
  const double horizon = std::max(config.orca_time_horizon, 1e-3);
  Point2 unit_w;
  Point2 u;
  if (distance_squared <= radius_squared) {
    const double inv_step = 1.0/std::max(config.orca_time_step, 1e-3);
    const Point2 w = relative_velocity-inv_step*position;
    const double w_length = std::hypot(w.x, w.y);
    if (w_length > 1e-9) unit_w = {w.x/w_length, w.y/w_length};
    else unit_w = {-radial.x, -radial.y};
    u = (combined_radius*inv_step-w_length)*unit_w;
  } else {
    const double inv_horizon = 1.0/horizon;
    const Point2 w = relative_velocity-inv_horizon*position;
    const double w_length_squared = normSquared(w);
    const double dot_product = dot(w, position);
    if (dot_product < 0.0 && dot_product*dot_product >
        radius_squared*w_length_squared && w_length_squared > 1e-12) {
      const double w_length = std::sqrt(w_length_squared);
      unit_w = {w.x/w_length, w.y/w_length};
      u = (combined_radius*inv_horizon-w_length)*unit_w;
    } else {
      const double denominator = distance_squared-radius_squared;
      const double cross_value = position.x*relative_velocity.y-
                                 position.y*relative_velocity.x;
      const double a = distance_squared;
      const double b = dot(position, relative_velocity);
      const double c = normSquared(relative_velocity)-
          cross_value*cross_value/std::max(denominator, 1e-12);
      const double discriminant = std::max(0.0, b*b-a*c);
      const double t = (b+std::sqrt(discriminant))/a;
      const Point2 cone_w = relative_velocity-t*position;
      const double cone_length = std::hypot(cone_w.x, cone_w.y);
      if (cone_length <= 1e-9) return false;
      unit_w = {cone_w.x/cone_length, cone_w.y/cone_length};
      u = (combined_radius*t-cone_length)*unit_w;
    }
  }
  const Point2 plane_point = agent_velocity+u;
  // ORCA acts only when the preferred velocity lies in the velocity obstacle;
  // this prevents unnecessary lateral motion around distant pedestrians.
  if (dot(preferred-plane_point, unit_w) >= -1e-9) return false;
  result->nx = -unit_w.x;
  result->ny = -unit_w.y;
  result->limit = -dot(unit_w, plane_point);
  return true;
}

}  // namespace

ObstacleSafetyFilter::ObstacleSafetyFilter(const ObstacleFilterConfig& config)
    : config_(config) {
  if (!(config_.vehicle_radius > 0.0) || !(config_.safety_margin >= 0.0) ||
      !(config_.dynamic_safety_margin >= config_.safety_margin) ||
      !(config_.effective_deceleration > 0.0) || !(config_.latency_budget >= 0.0) ||
      !(config_.min_obstacle_z < config_.max_obstacle_z) ||
      !(config_.max_obstacle_range > config_.vehicle_radius + config_.safety_margin) ||
      !(config_.max_horizontal_speed > 0.0) ||
      !(config_.escape_speed_threshold >= 0.0) ||
      !(config_.dynamic_confidence_scale >= 0.0) ||
      !(config_.dynamic_uncertainty_velocity_cap >= 0.0) ||
      !(config_.orca_time_horizon > 0.0) || !(config_.orca_time_step > 0.0) ||
      !(config_.orca_safety_distance >= 0.0) ||
      !(config_.dynamic_obstacle_radius >= 0.0) ||
      !(config_.dynamic_lateral_escape_speed >= 0.0) ||
      config_.escape_speed_threshold > config_.max_horizontal_speed) {
    throw std::invalid_argument("invalid obstacle-safety-filter configuration");
  }
}

bool ObstacleSafetyFilter::finite(double value) { return std::isfinite(value); }

bool ObstacleSafetyFilter::finite(const Vector3& value) {
  return finite(value.x) && finite(value.y) && finite(value.z);
}

ObstacleFilterOutput ObstacleSafetyFilter::filter(
    const Vector3& preferred_velocity_body,
    const Vector3& actual_velocity_body,
    const std::vector<ObstaclePoint>& obstacles,
    const Vector3& goal_direction_body) const {
  if (!finite(preferred_velocity_body) || !finite(actual_velocity_body) ||
      !finite(goal_direction_body)) {
    throw std::invalid_argument("filter velocity contains NaN or Inf");
  }

  ObstacleFilterOutput output;
  output.safe_velocity_body = preferred_velocity_body;
  const double preferred_speed = std::hypot(
      output.safe_velocity_body.x, output.safe_velocity_body.y);
  if (preferred_speed > config_.max_horizontal_speed) {
    const double scale = config_.max_horizontal_speed/preferred_speed;
    output.safe_velocity_body.x *= scale;
    output.safe_velocity_body.y *= scale;
  }
  output.minimum_surface_clearance = std::numeric_limits<double>::infinity();
  output.minimum_allowed_closing_speed = std::numeric_limits<double>::infinity();

  std::vector<HalfPlane> constraints;
  constraints.reserve(obstacles.size());
  const Point2 preferred{output.safe_velocity_body.x, output.safe_velocity_body.y};
  Point2 solver_preferred = preferred;
  double nearest_dynamic_distance = std::numeric_limits<double>::infinity();
  double nearest_dynamic_side = 0.0;

  for (const auto& point : obstacles) {
    if (!finite(point.x) || !finite(point.y) || !finite(point.z) ||
        point.z < config_.min_obstacle_z || point.z > config_.max_obstacle_z) {
      continue;
    }
    const double distance = std::hypot(point.x, point.y);
    if (!(distance > 1e-6) || distance > config_.max_obstacle_range) continue;

    const double unit_x = point.x/distance;
    const double unit_y = point.y/distance;
    const double surface_clearance = distance-config_.vehicle_radius;
    output.minimum_surface_clearance =
        std::min(output.minimum_surface_clearance, surface_clearance);

    if (point.dynamic && point.x > 0.0 &&
        distance < nearest_dynamic_distance) {
      nearest_dynamic_distance = distance;
      // Select the side away from the observed target. If it is almost
      // centered, use a deterministic positive-y tie break.
      nearest_dynamic_side = point.y < -0.10 ? 1.0 :
          (point.y > 0.10 ? -1.0 : 1.0);
    }

    if (point.dynamic && config_.use_orca_for_dynamic) {
      if (!finite(point.relative_radial_velocity) ||
          !finite(point.relative_tangential_velocity) ||
          !finite(point.radial_velocity_uncertainty) ||
          point.radial_velocity_uncertainty < 0.0) {
        continue;
      }
      HalfPlane orca;
      const Point2 actual{actual_velocity_body.x, actual_velocity_body.y};
      if (std::abs(point.relative_tangential_velocity) > 1e-3 &&
          makeOrcaConstraint(point, actual, config_, preferred, &orca)) {
        constraints.push_back(orca);
        // A valid ORCA plane already contains the moving target's full
        // relative velocity. Do not append the legacy radial retreat plane
        // for this target, otherwise it suppresses the lateral escape that
        // ORCA is intended to preserve.
        ++output.dynamic_constraints;
        continue;
      }
    }

    const double safety_margin = point.dynamic ?
        config_.dynamic_safety_margin : config_.safety_margin;
    const double available = std::max(0.0, surface_clearance-safety_margin);
    const double a = config_.effective_deceleration;
    const double latency = config_.latency_budget;
    const double allowed = std::max(
        0.0, -a*latency + std::sqrt(a*a*latency*latency + 2.0*a*available));
    output.minimum_allowed_closing_speed =
        std::min(output.minimum_allowed_closing_speed, allowed);

    // The admissible closing speed includes the measured actuator response:
    // v*latency + v^2/(2*a_effective) <= available clearance.
    const double actual_ego_radial =
        actual_velocity_body.x*unit_x + actual_velocity_body.y*unit_y;
    double obstacle_radial = 0.0;
    double actual_closing = std::max(0.0, actual_ego_radial);
    if (point.dynamic) {
      if (!finite(point.relative_radial_velocity) ||
          !finite(point.radial_velocity_uncertainty) ||
          point.radial_velocity_uncertainty < 0.0) {
        continue;
      }
      // The perception channel reports dr/dt = v_obstacle - v_ego.
      // Subtract uncertainty to bias toward a faster approaching obstacle,
      // then recover its radial world velocity in the current body frame.
      const double uncertainty_allowance = std::min(
          config_.dynamic_uncertainty_velocity_cap,
          config_.dynamic_confidence_scale*point.radial_velocity_uncertainty);
      const double robust_relative_radial =
          point.relative_radial_velocity-uncertainty_allowance;
      // Uncertainty makes the braking-envelope test trigger earlier, but it
      // must not create an impossible retreat velocity. The actual half-plane
      // therefore follows the nominal measured obstacle velocity.
      obstacle_radial = point.relative_radial_velocity+actual_ego_radial;
      actual_closing = std::max(0.0, -robust_relative_radial);
      ++output.dynamic_constraints;
    }
    const double actual_stopping_requirement =
        latency*actual_closing + actual_closing*actual_closing/(2.0*a);
    // n.v_ego <= n.v_obstacle + allowed_relative_closing. If already inside
    // the measured relative braking envelope, demand non-closing motion.
    const double admissible_closing = std::max(
        -config_.max_horizontal_speed,
        obstacle_radial+(actual_stopping_requirement > available ? 0.0 : allowed));
    constraints.push_back({unit_x, unit_y, admissible_closing});
  }

  constexpr double kConstraintTolerance = 1e-10;
  for (const auto& constraint : constraints) {
    if (signedViolation(constraint, preferred) > kConstraintTolerance) {
      ++output.active_constraints;
    }
  }

  const double goal_norm = std::hypot(goal_direction_body.x, goal_direction_body.y);
  if (preferred.x > 0.10 && goal_direction_body.x > 0.20 &&
      std::abs(goal_direction_body.y) < 0.80 &&
      nearest_dynamic_distance < config_.orca_time_horizon +
          config_.dynamic_obstacle_radius &&
      config_.dynamic_lateral_escape_speed > 0.0) {
    solver_preferred.y += nearest_dynamic_side *
        config_.dynamic_lateral_escape_speed;
  }
  const bool stalled_with_goal = preferred_speed < config_.escape_speed_threshold &&
                                 goal_norm > 1e-9;
  if (output.active_constraints > 0 || stalled_with_goal) {
    // Solve all half-planes together inside the circular speed envelope. This
    // is the same bounded velocity program used by the ROS1 ORCA shield.
    const double bound = config_.max_horizontal_speed;
    Point2 projected;
    if (!closestPointInDisk(solver_preferred, constraints, bound, &projected)) {
      // A zero command is unsafe when a moving obstacle can reach a hovering
      // vehicle. If no velocity satisfies every half-plane, use the bounded
      // velocity that minimizes the worst and aggregate constraint violation.
      const Point2 rescue = leastViolatingVelocity(preferred, constraints, bound);
      output.safe_velocity_body.x = rescue.x;
      output.safe_velocity_body.y = rescue.y;
      output.escape_assist = true;
    } else if (output.active_constraints > 0) {
      output.safe_velocity_body.x = projected.x;
      output.safe_velocity_body.y = projected.y;
    }
    output.active = output.active_constraints > 0;

    const double projected_speed = std::hypot(
        output.safe_velocity_body.x, output.safe_velocity_body.y);
    if (closestPointInDisk({0.0, 0.0}, constraints, bound, &projected) &&
        projected_speed < config_.escape_speed_threshold &&
        goal_norm > 1e-9) {
      // A pure control-barrier projection legitimately chooses zero when the
      // desired velocity points straight into an obstacle. A Beta-mean policy
      // can also collapse near zero in an ambiguous dense observation. In both
      // cases choose the feasible velocity with greatest goal progress.
      const Point2 goal_velocity{
          config_.max_horizontal_speed*goal_direction_body.x/goal_norm,
          config_.max_horizontal_speed*goal_direction_body.y/goal_norm};
      Point2 best;
      const bool goal_velocity_feasible =
          maxProgressInDisk(goal_velocity, constraints, bound, &best);
      if (!goal_velocity_feasible) best = projected;
      const double best_speed = std::hypot(best.x, best.y);
      if (best_speed > config_.max_horizontal_speed) {
        best.x *= config_.max_horizontal_speed/best_speed;
        best.y *= config_.max_horizontal_speed/best_speed;
      }
      double best_progress = (goal_direction_body.x*best.x+
                              goal_direction_body.y*best.y)/goal_norm;
      if (best_progress > 1e-6) {
        output.safe_velocity_body.x = best.x;
        output.safe_velocity_body.y = best.y;
        output.escape_assist = true;
        output.active = true;
      }
    }
  }

  if (!std::isfinite(output.minimum_surface_clearance)) {
    output.minimum_surface_clearance = -1.0;
    output.minimum_allowed_closing_speed = -1.0;
  }
  return output;
}

}  // namespace dnav_controller
