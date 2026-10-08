#ifndef DNAV_CONTROLLER_DYNAMIC_OBSTACLE_CLUSTER_H
#define DNAV_CONTROLLER_DYNAMIC_OBSTACLE_CLUSTER_H

#include <cstddef>
#include <vector>

#include <dnav_controller/obstacle_safety_filter.h>

namespace dnav_controller {

struct DynamicGridSample {
  ObstaclePoint point;
  double distance = 0.0;
  double absolute_speed = 0.0;
  bool valid = false;
};

struct DynamicClusterConfig {
  int height = 6;
  int width = 36;
  int minimum_pixels = 2;
  int maximum_clusters = 3;
  double depth_tolerance = 0.75;
  double merge_distance = 1.0;
  double min_obstacle_z = -0.45;
  double max_obstacle_z = 0.25;
  double max_obstacle_range = 4.0;
  double min_absolute_speed = 0.15;
  double max_absolute_speed = 3.0;
};

struct DynamicTrackConfig {
  int maximum_tracks = 3;
  double association_distance = 1.0;
  double hold_time = 0.6;
  double update_alpha = 0.25;
};

class DynamicObstacleTracker {
 public:
  explicit DynamicObstacleTracker(const DynamicTrackConfig& config);
  void update(const std::vector<ObstaclePoint>& detections, double stamp);
  std::vector<ObstaclePoint> obstacles(double stamp) const;

 private:
  struct Track {
    ObstaclePoint point;
    double state_stamp;
    double detection_stamp;
  };
  ObstaclePoint predict(const Track& track, double stamp) const;

  DynamicTrackConfig config_;
  std::vector<Track> tracks_;
};

std::vector<ObstaclePoint> clusterDynamicObstacles(
    const std::vector<DynamicGridSample>& samples,
    const DynamicClusterConfig& config);

}  // namespace dnav_controller

#endif
