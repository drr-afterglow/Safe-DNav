#include <dnav_controller/dynamic_obstacle_cluster.h>

#include <algorithm>
#include <cmath>
#include <queue>
#include <stdexcept>

namespace dnav_controller {

namespace {

double median(std::vector<double> values) {
  const std::size_t middle = values.size()/2;
  std::nth_element(values.begin(), values.begin()+middle, values.end());
  if (values.size()%2 != 0) return values[middle];
  const double upper = values[middle];
  std::nth_element(values.begin(), values.begin()+middle-1, values.end());
  return 0.5*(values[middle-1]+upper);
}

bool finiteSample(const DynamicGridSample& sample) {
  const auto& point = sample.point;
  return sample.valid && std::isfinite(sample.distance) && sample.distance > 0.0 &&
      std::isfinite(sample.absolute_speed) &&
      std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z) &&
      std::isfinite(point.relative_radial_velocity) &&
      std::isfinite(point.relative_tangential_velocity) &&
      std::isfinite(point.radial_velocity_uncertainty) &&
      point.radial_velocity_uncertainty >= 0.0;
}

}  // namespace

DynamicObstacleTracker::DynamicObstacleTracker(const DynamicTrackConfig& config)
    : config_(config) {
  if (config_.maximum_tracks <= 0 ||
      !(config_.association_distance > 0.0) ||
      !(config_.hold_time >= 0.0) || !(config_.update_alpha > 0.0) ||
      !(config_.update_alpha <= 1.0)) {
    throw std::invalid_argument("invalid dynamic tracker configuration");
  }
}

ObstaclePoint DynamicObstacleTracker::predict(const Track& track,
                                               double stamp) const {
  ObstaclePoint predicted = track.point;
  const double dt = std::max(0.0, stamp-track.state_stamp);
  const double distance = std::hypot(predicted.x, predicted.y);
  if (distance > 1e-6 && dt > 0.0) {
    const double predicted_distance = std::max(
        0.05, distance+predicted.relative_radial_velocity*dt);
    const double scale = predicted_distance/distance;
    predicted.x *= scale;
    predicted.y *= scale;
  }
  return predicted;
}

void DynamicObstacleTracker::update(
    const std::vector<ObstaclePoint>& detections, double stamp) {
  if (!std::isfinite(stamp)) {
    throw std::invalid_argument("dynamic tracker stamp is not finite");
  }
  std::vector<Track> retained;
  retained.reserve(tracks_.size()+detections.size());
  for (const auto& track : tracks_) {
    if (stamp+1e-9 < track.state_stamp) {
      throw std::invalid_argument("dynamic tracker stamp moved backwards");
    }
    if (stamp-track.detection_stamp <= config_.hold_time) {
      retained.push_back({predict(track, stamp), stamp, track.detection_stamp});
    }
  }
  tracks_.swap(retained);

  std::vector<bool> matched(tracks_.size(), false);
  for (const auto& detection : detections) {
    if (!detection.dynamic || !std::isfinite(detection.x) ||
        !std::isfinite(detection.y) ||
        !std::isfinite(detection.relative_radial_velocity)) {
      continue;
    }
    std::size_t best = tracks_.size();
    double best_distance = config_.association_distance;
    for (std::size_t index=0; index<tracks_.size(); ++index) {
      if (matched[index]) continue;
      const double distance = std::hypot(
          detection.x-tracks_[index].point.x,
          detection.y-tracks_[index].point.y);
      if (distance <= best_distance) {
        best = index;
        best_distance = distance;
      }
    }
    if (best < tracks_.size()) {
      const double alpha = config_.update_alpha;
      const auto& previous = tracks_[best].point;
      ObstaclePoint smoothed = detection;
      smoothed.x = (1.0-alpha)*previous.x+alpha*detection.x;
      smoothed.y = (1.0-alpha)*previous.y+alpha*detection.y;
      smoothed.z = (1.0-alpha)*previous.z+alpha*detection.z;
      smoothed.relative_radial_velocity =
          (1.0-alpha)*previous.relative_radial_velocity+
          alpha*detection.relative_radial_velocity;
      smoothed.relative_tangential_velocity =
          (1.0-alpha)*previous.relative_tangential_velocity+
          alpha*detection.relative_tangential_velocity;
      smoothed.radial_velocity_uncertainty =
          (1.0-alpha)*previous.radial_velocity_uncertainty+
          alpha*detection.radial_velocity_uncertainty;
      tracks_[best] = {smoothed, stamp, stamp};
      matched[best] = true;
    } else {
      tracks_.push_back({detection, stamp, stamp});
      matched.push_back(true);
    }
  }
  std::sort(tracks_.begin(), tracks_.end(), [](const Track& first,
                                               const Track& second) {
    return std::hypot(first.point.x, first.point.y) <
           std::hypot(second.point.x, second.point.y);
  });
  if (tracks_.size() > static_cast<std::size_t>(config_.maximum_tracks)) {
    tracks_.resize(config_.maximum_tracks);
  }
}

std::vector<ObstaclePoint> DynamicObstacleTracker::obstacles(double stamp) const {
  if (!std::isfinite(stamp)) {
    throw std::invalid_argument("dynamic tracker stamp is not finite");
  }
  std::vector<ObstaclePoint> result;
  for (const auto& track : tracks_) {
    if (stamp+1e-9 < track.state_stamp) {
      throw std::invalid_argument("dynamic tracker stamp moved backwards");
    }
    if (stamp-track.detection_stamp <= config_.hold_time) {
      result.push_back(predict(track, stamp));
    }
  }
  return result;
}

std::vector<ObstaclePoint> clusterDynamicObstacles(
    const std::vector<DynamicGridSample>& samples,
    const DynamicClusterConfig& config) {
  if (config.height <= 0 || config.width <= 0 || config.minimum_pixels <= 0 ||
      config.maximum_clusters <= 0 || !(config.depth_tolerance >= 0.0) ||
      !(config.merge_distance >= 0.0) ||
      !(config.min_obstacle_z < config.max_obstacle_z) ||
      !(config.max_obstacle_range > 0.0) ||
      !(config.min_absolute_speed >= 0.0) ||
      !(config.max_absolute_speed > config.min_absolute_speed) ||
      samples.size() != static_cast<std::size_t>(config.height*config.width)) {
    throw std::invalid_argument("invalid dynamic clustering configuration");
  }

  std::vector<bool> visited(samples.size(), false);
  std::vector<std::pair<double, ObstaclePoint>> clustered;
  for (int seed=0; seed<config.height*config.width; ++seed) {
    if (visited[seed] || !finiteSample(samples[seed]) ||
        samples[seed].absolute_speed < config.min_absolute_speed ||
        samples[seed].absolute_speed > config.max_absolute_speed ||
        samples[seed].distance > config.max_obstacle_range) continue;
    std::queue<int> pending;
    std::vector<int> component;
    visited[seed] = true;
    pending.push(seed);
    while (!pending.empty()) {
      const int current = pending.front();
      pending.pop();
      component.push_back(current);
      const int row = current/config.width;
      const int column = current%config.width;
      for (int dr=-1; dr<=1; ++dr) {
        for (int dc=-1; dc<=1; ++dc) {
          if (dr == 0 && dc == 0) continue;
          const int next_row = row+dr;
          if (next_row < 0 || next_row >= config.height) continue;
          const int next_column = (column+dc+config.width)%config.width;
          const int next = next_row*config.width+next_column;
          if (visited[next] || !finiteSample(samples[next]) ||
              samples[next].absolute_speed < config.min_absolute_speed ||
              samples[next].absolute_speed > config.max_absolute_speed ||
              samples[next].distance > config.max_obstacle_range) continue;
          if (std::abs(samples[next].distance-samples[current].distance) >
              config.depth_tolerance) continue;
          visited[next] = true;
          pending.push(next);
        }
      }
    }
    if (static_cast<int>(component.size()) < config.minimum_pixels) continue;

    int representative = -1;
    double nearest = config.max_obstacle_range+1.0;
    std::vector<double> x_positions;
    std::vector<double> y_positions;
    std::vector<double> z_positions;
    std::vector<double> radial_velocities;
    std::vector<double> tangential_velocities;
    std::vector<double> uncertainties;
    x_positions.reserve(component.size());
    y_positions.reserve(component.size());
    z_positions.reserve(component.size());
    radial_velocities.reserve(component.size());
    tangential_velocities.reserve(component.size());
    uncertainties.reserve(component.size());
    for (const int index : component) {
      const auto& sample = samples[index];
      x_positions.push_back(sample.point.x);
      y_positions.push_back(sample.point.y);
      z_positions.push_back(sample.point.z);
      radial_velocities.push_back(sample.point.relative_radial_velocity);
      tangential_velocities.push_back(sample.point.relative_tangential_velocity);
      uncertainties.push_back(sample.point.radial_velocity_uncertainty);
      if (sample.point.z >= config.min_obstacle_z &&
          sample.point.z <= config.max_obstacle_z && sample.distance < nearest) {
        representative = index;
        nearest = sample.distance;
      }
    }
    if (representative < 0) continue;
    ObstaclePoint obstacle = samples[representative].point;
    // A nearest return is a surface sample, not an estimate of the mover's
    // center.  Using it as the ORCA center makes the half-plane jump whenever
    // the depth network changes which body pixel is nearest.  The coordinate
    // median is robust to missing pixels and keeps the inflated obstacle
    // geometry centered on the observed cluster.
    obstacle.x = median(x_positions);
    obstacle.y = median(y_positions);
    obstacle.z = median(z_positions);
    obstacle.relative_radial_velocity = median(radial_velocities);
    obstacle.relative_tangential_velocity = median(tangential_velocities);
    obstacle.radial_velocity_uncertainty = median(uncertainties);
    obstacle.dynamic = true;
    clustered.emplace_back(nearest, obstacle);
  }

  std::sort(clustered.begin(), clustered.end(),
            [](const auto& first, const auto& second) { return first.first < second.first; });
  // A single person can be split into several connected components by a
  // depth discontinuity or one missed network pixel. Merge nearby components
  // before enforcing the object-count limit, otherwise duplicate half-planes
  // can exclude every useful avoidance velocity.
  std::vector<std::pair<double, ObstaclePoint>> merged;
  for (const auto& candidate : clustered) {
    auto match = merged.end();
    for (auto it=merged.begin(); it!=merged.end(); ++it) {
      if (std::hypot(candidate.second.x-it->second.x,
                     candidate.second.y-it->second.y) <= config.merge_distance) {
        match = it;
        break;
      }
    }
    if (match == merged.end()) {
      merged.push_back(candidate);
      continue;
    }
    const double robust_radial = std::min(
        match->second.relative_radial_velocity,
        candidate.second.relative_radial_velocity);
    // Keep the median-like tangential estimate from the nearest component;
    // unlike radial closing speed, its sign is meaningful for ORCA escape.
    const double tangential = candidate.first < match->first
        ? candidate.second.relative_tangential_velocity
        : match->second.relative_tangential_velocity;
    const double robust_uncertainty = std::max(
        match->second.radial_velocity_uncertainty,
        candidate.second.radial_velocity_uncertainty);
    if (candidate.first < match->first) *match = candidate;
    match->second.relative_radial_velocity = robust_radial;
    match->second.relative_tangential_velocity = tangential;
    match->second.radial_velocity_uncertainty = robust_uncertainty;
  }
  if (merged.size() > static_cast<std::size_t>(config.maximum_clusters)) {
    merged.resize(config.maximum_clusters);
  }
  std::vector<ObstaclePoint> result;
  result.reserve(merged.size());
  for (const auto& item : merged) result.push_back(item.second);
  return result;
}

}  // namespace dnav_controller
