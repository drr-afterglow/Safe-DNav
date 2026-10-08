#include <algorithm>
#include <cmath>
#include <memory>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <diagnostic_msgs/DiagnosticArray.h>
#include <diagnostic_msgs/DiagnosticStatus.h>
#include <diagnostic_msgs/KeyValue.h>
#include <geometry_msgs/Vector3Stamped.h>
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Odometry.h>
#include <ros/ros.h>
#include <sensor_msgs/LaserScan.h>

#include <dnav_controller/obstacle_safety_filter.h>
#include <dnav_controller/dynamic_obstacle_cluster.h>
#include <dnav_perception/PerceptionObservation.h>

namespace dc = dnav_controller;

namespace {
diagnostic_msgs::KeyValue keyValue(const std::string& key, const std::string& value) {
  diagnostic_msgs::KeyValue result;
  result.key = key;
  result.value = value;
  return result;
}

template <typename T>
std::string asString(const T& value) {
  std::ostringstream stream;
  stream << value;
  return stream.str();
}
}  // namespace

class ObstacleSafetyFilterNode {
 public:
  ObstacleSafetyFilterNode() : nh_(), pnh_("~") {
    dc::ObstacleFilterConfig config;
    config.vehicle_radius = parameter("vehicle_radius", 0.30);
    config.safety_margin = parameter("safety_margin", 0.50);
    config.dynamic_safety_margin = parameter("dynamic_safety_margin", 0.90);
    config.effective_deceleration = parameter("effective_deceleration", 0.25);
    config.latency_budget = parameter("latency_budget", 0.15);
    config.min_obstacle_z = parameter("min_obstacle_z", -0.45);
    config.max_obstacle_z = parameter("max_obstacle_z", 0.25);
    config.max_obstacle_range = parameter("max_obstacle_range", 4.0);
    config.max_horizontal_speed = parameter("max_horizontal_speed", 0.5);
    config.escape_speed_threshold = parameter("escape_speed_threshold", 0.05);
    config.dynamic_confidence_scale = parameter("dynamic_confidence_scale", 2.0);
    config.dynamic_uncertainty_velocity_cap =
        parameter("dynamic_uncertainty_velocity_cap", 0.30);
    config.use_orca_for_dynamic = parameter("use_orca_for_dynamic", true);
    config.orca_time_horizon = parameter("orca_time_horizon", 2.0);
    config.orca_time_step = parameter("orca_time_step", 0.05);
    config.orca_safety_distance = parameter("orca_safety_distance", 0.30);
    config.dynamic_obstacle_radius = parameter("dynamic_obstacle_radius", 0.45);
    config.dynamic_lateral_escape_speed =
        parameter("dynamic_lateral_escape_speed", 0.35);
    filter_.reset(new dc::ObstacleSafetyFilter(config));

    rate_ = parameter("filter_rate", 50.0);
    policy_timeout_ = parameter("policy_timeout", 0.10);
    odometry_timeout_ = parameter("odometry_timeout", 0.10);
    scan_timeout_ = parameter("scan_timeout", 0.20);
    perception_timeout_ = parameter("perception_timeout", 0.25);
    future_tolerance_ = parameter("future_stamp_tolerance", 0.05);
    scan_height_ = parameter("scan_height", 16);
    scan_width_ = parameter("scan_width", 360);
    vertical_min_ = parameter("vertical_min_deg", -7.0)*M_PI/180.0;
    vertical_max_ = parameter("vertical_max_deg", 52.0)*M_PI/180.0;
    dynamic_max_range_ = parameter("dynamic_max_range", 10.0);
    dynamic_velocity_scale_ = parameter("dynamic_velocity_scale", 5.0);
    dynamic_uncertainty_scale_ = parameter("dynamic_uncertainty_scale", 10.0);
    dynamic_valid_depth_threshold_ = parameter("dynamic_valid_depth_threshold", 0.99);
    dynamic_cluster_config_.height = 6;
    dynamic_cluster_config_.width = 36;
    dynamic_cluster_config_.minimum_pixels = parameter("dynamic_cluster_minimum_pixels", 2);
    dynamic_cluster_config_.maximum_clusters = parameter("dynamic_maximum_clusters", 3);
    dynamic_cluster_config_.depth_tolerance = parameter("dynamic_cluster_depth_tolerance", 0.75);
    dynamic_cluster_config_.merge_distance = parameter("dynamic_cluster_merge_distance", 1.0);
    dynamic_cluster_config_.min_obstacle_z = config.min_obstacle_z;
    dynamic_cluster_config_.max_obstacle_z = config.max_obstacle_z;
    dynamic_cluster_config_.max_obstacle_range = config.max_obstacle_range;
    dynamic_cluster_config_.min_absolute_speed = parameter("dynamic_min_absolute_speed", 0.15);
    dynamic_cluster_config_.max_absolute_speed = parameter("dynamic_max_absolute_speed", 3.0);
    dc::DynamicTrackConfig track_config;
    track_config.maximum_tracks = dynamic_cluster_config_.maximum_clusters;
    track_config.association_distance = parameter("dynamic_track_association_distance", 1.0);
    track_config.hold_time = parameter("dynamic_track_hold_time", 0.6);
    track_config.update_alpha = parameter("dynamic_track_update_alpha", 0.25);
    dynamic_tracker_.reset(new dc::DynamicObstacleTracker(track_config));
    world_frame_ = parameter<std::string>("world_frame", "map");
    body_frame_ = parameter<std::string>("body_frame", "base_link");
    sensor_frame_ = parameter<std::string>("sensor_frame", "mid360_link");
    if (!(rate_ > 0.0) || !(policy_timeout_ > 0.0) || !(odometry_timeout_ > 0.0) ||
        !(scan_timeout_ > 0.0) || !(perception_timeout_ > 0.0) ||
        !(future_tolerance_ >= 0.0) || !(dynamic_max_range_ > 0.0) ||
        !(dynamic_velocity_scale_ > 0.0) || !(dynamic_uncertainty_scale_ >= 0.0) ||
        !(dynamic_valid_depth_threshold_ > 0.0) ||
        !(dynamic_valid_depth_threshold_ <= 1.0) ||
        dynamic_cluster_config_.minimum_pixels <= 0 ||
        dynamic_cluster_config_.maximum_clusters <= 0 ||
        !(dynamic_cluster_config_.depth_tolerance >= 0.0) ||
        !(dynamic_cluster_config_.merge_distance >= 0.0) ||
        !(dynamic_cluster_config_.min_absolute_speed >= 0.0) ||
        !(dynamic_cluster_config_.max_absolute_speed >
          dynamic_cluster_config_.min_absolute_speed) ||
        scan_height_ <= 0 || scan_width_ <= 0 || !(vertical_min_ < vertical_max_)) {
      throw std::invalid_argument("invalid obstacle-safety-filter runtime configuration");
    }

    output_pub_ = nh_.advertise<geometry_msgs::Vector3Stamped>(
        parameter<std::string>("output_topic", "/dnav/safety/world_velocity"), 1);
    diagnostics_pub_ = nh_.advertise<diagnostic_msgs::DiagnosticArray>(
        parameter<std::string>("diagnostics_topic", "/dnav/safety/diagnostics"), 1);
    policy_sub_ = nh_.subscribe(
        parameter<std::string>("policy_topic", "/dnav/policy/world_velocity"), 1,
        &ObstacleSafetyFilterNode::policy, this);
    goal_sub_ = nh_.subscribe(
        parameter<std::string>("goal_topic", "/dnav/policy/goal"), 1,
        &ObstacleSafetyFilterNode::goal, this);
    odometry_sub_ = nh_.subscribe(
        parameter<std::string>("odometry_topic", "/dnav/uav/odom_raw"), 20,
        &ObstacleSafetyFilterNode::odometry, this);
    scan_sub_ = nh_.subscribe(
        parameter<std::string>("scan_topic", "/dnav/uav/mid360/scan"), 1,
        &ObstacleSafetyFilterNode::scan, this);
    perception_sub_ = nh_.subscribe(
        parameter<std::string>("perception_topic", "/dnav/perception/observation"), 1,
        &ObstacleSafetyFilterNode::perception, this);
    timer_ = nh_.createTimer(ros::Duration(1.0/rate_), &ObstacleSafetyFilterNode::tick, this);
    ROS_INFO("braking-aware obstacle safety filter ready at %.1f Hz", rate_);
  }

 private:
  template <typename T>
  T parameter(const std::string& name, const T& fallback) {
    T value;
    pnh_.param<T>(name, value, fallback);
    return value;
  }

  bool fresh(const ros::Time& now, const ros::Time& stamp, double timeout) const {
    if (stamp.isZero()) return false;
    const double age = (now-stamp).toSec();
    return age >= -future_tolerance_ && age <= timeout;
  }

  void policy(const geometry_msgs::Vector3Stamped::ConstPtr& message) {
    if (message->header.frame_id != world_frame_ || message->header.stamp.isZero() ||
        !std::isfinite(message->vector.x) || !std::isfinite(message->vector.y) ||
        !std::isfinite(message->vector.z)) {
      ++invalid_input_count_;
      return;
    }
    policy_ = *message;
    have_policy_ = true;
  }

  void goal(const geometry_msgs::PoseStamped::ConstPtr& message) {
    const auto& position = message->pose.position;
    if (message->header.frame_id != world_frame_ ||
        !std::isfinite(position.x) || !std::isfinite(position.y) ||
        !std::isfinite(position.z)) {
      ++invalid_input_count_;
      return;
    }
    goal_ = *message;
    have_goal_ = true;
  }

  void odometry(const nav_msgs::Odometry::ConstPtr& message) {
    const auto& q = message->pose.pose.orientation;
    const auto& velocity = message->twist.twist.linear;
    const double norm = std::sqrt(q.w*q.w+q.x*q.x+q.y*q.y+q.z*q.z);
    if (message->header.frame_id != world_frame_ || message->child_frame_id != body_frame_ ||
        message->header.stamp.isZero() || norm < 1e-6 || !std::isfinite(norm) ||
        !std::isfinite(velocity.x) || !std::isfinite(velocity.y) ||
        !std::isfinite(velocity.z)) {
      ++invalid_input_count_;
      return;
    }
    odometry_ = *message;
    const double w=q.w/norm, x=q.x/norm, y=q.y/norm, z=q.z/norm;
    yaw_ = std::atan2(2.0*(w*z+x*y), 1.0-2.0*(y*y+z*z));
    have_odometry_ = std::isfinite(yaw_);
  }

  void scan(const sensor_msgs::LaserScan::ConstPtr& message) {
    if (message->header.frame_id != sensor_frame_ || message->header.stamp.isZero() ||
        message->ranges.size() != static_cast<std::size_t>(scan_height_*scan_width_) ||
        !std::isfinite(message->angle_min) || !std::isfinite(message->angle_increment) ||
        !(message->angle_increment > 0.0)) {
      ++invalid_input_count_;
      return;
    }
    obstacles_.clear();
    obstacles_.reserve(scan_width_);
    const auto& config = filter_->config();
    for (int column=0; column<scan_width_; ++column) {
      double best_range = std::numeric_limits<double>::infinity();
      double best_pitch = 0.0;
      for (int row=0; row<scan_height_; ++row) {
        const double range = message->ranges[row*scan_width_+column];
        if (!std::isfinite(range) || range < message->range_min ||
            range > message->range_max || range > config.max_obstacle_range) continue;
        const double ratio = scan_height_ == 1 ? 0.0 :
            static_cast<double>(row)/(scan_height_-1);
        const double pitch = vertical_min_ + ratio*(vertical_max_-vertical_min_);
        const double z = range*std::sin(pitch);
        if (z < config.min_obstacle_z || z > config.max_obstacle_z) continue;
        if (range < best_range) {
          best_range = range;
          best_pitch = pitch;
        }
      }
      if (!std::isfinite(best_range)) continue;
      const double yaw = message->angle_min + column*message->angle_increment;
      const double horizontal = best_range*std::cos(best_pitch);
      obstacles_.push_back({horizontal*std::cos(yaw), horizontal*std::sin(yaw),
                            best_range*std::sin(best_pitch)});
    }
    scan_stamp_ = message->header.stamp;
    have_scan_ = true;
  }

  void perception(const dnav_perception::PerceptionObservation::ConstPtr& message) {
    constexpr int kHeight = 6;
    constexpr int kWidth = 36;
    constexpr int kPixels = kHeight*kWidth;
    if (message->header.frame_id != body_frame_ || message->header.stamp.isZero()) {
      ++invalid_input_count_;
      return;
    }
    std::vector<dc::ObstaclePoint> detected_obstacles;
    if (message->history_valid && have_odometry_) {
      std::vector<dc::DynamicGridSample> samples(kPixels);
      const auto& ego = odometry_.twist.twist.linear;
      for (int row=0; row<kHeight; ++row) {
        const double pitch_ratio = (static_cast<double>(row)+0.5)/kHeight;
        const double pitch = vertical_max_-pitch_ratio*(vertical_max_-vertical_min_);
        for (int column=0; column<kWidth; ++column) {
          const int pixel = row*kWidth+column;
          const double normalized_depth = message->lidar_dynamic[pixel];
          const double normalized_radial = message->lidar_dynamic[kPixels+pixel];
          const double normalized_tangential = message->lidar_dynamic[2*kPixels+pixel];
          const double normalized_elevation = message->lidar_dynamic[3*kPixels+pixel];
          const double normalized_sigma = message->lidar_dynamic[4*kPixels+pixel];
          if (!std::isfinite(normalized_depth) ||
              !std::isfinite(normalized_radial) ||
              !std::isfinite(normalized_tangential) ||
              !std::isfinite(normalized_elevation) ||
              !std::isfinite(normalized_sigma) ||
              normalized_depth < 0.0 ||
              normalized_depth >= dynamic_valid_depth_threshold_ ||
              normalized_sigma < 0.0) {
            continue;
          }
          const double distance = normalized_depth*dynamic_max_range_;
          if (!(distance > 1e-6)) continue;
          const double yaw = -M_PI+(static_cast<double>(column)+0.5)*2.0*M_PI/kWidth;
          const double cos_pitch = std::cos(pitch);
          const double sin_pitch = std::sin(pitch);
          const double cos_yaw = std::cos(yaw);
          const double sin_yaw = std::sin(yaw);
          const double horizontal = distance*std::cos(pitch);
          const double z = distance*std::sin(pitch);
          const double ego_radial = ego.x*cos_pitch*cos_yaw+
              ego.y*cos_pitch*sin_yaw+ego.z*sin_pitch;
          const double ego_tangential = -ego.x*sin_yaw+ego.y*cos_yaw;
          const double ego_elevation = -ego.x*sin_pitch*cos_yaw-
              ego.y*sin_pitch*sin_yaw+ego.z*cos_pitch;
          const double absolute_radial = normalized_radial*dynamic_velocity_scale_+
              ego_radial;
          const double absolute_tangential =
              normalized_tangential*dynamic_velocity_scale_+ego_tangential;
          const double absolute_elevation =
              normalized_elevation*dynamic_velocity_scale_+ego_elevation;
          const double absolute_speed = std::sqrt(
              absolute_radial*absolute_radial+
              absolute_tangential*absolute_tangential+
              absolute_elevation*absolute_elevation);
          samples[pixel] = {{horizontal*cos_yaw, horizontal*sin_yaw, z,
                             normalized_radial*dynamic_velocity_scale_,
                             normalized_sigma*dynamic_uncertainty_scale_, true,
                             normalized_tangential*dynamic_velocity_scale_},
                            distance, absolute_speed, true};
        }
      }
      detected_obstacles = dc::clusterDynamicObstacles(samples, dynamic_cluster_config_);
    }
    dynamic_tracker_->update(detected_obstacles, message->header.stamp.toSec());
    perception_stamp_ = message->header.stamp;
    have_perception_ = true;
  }

  void publishZero(const ros::Time& now, const std::string& reason) {
    geometry_msgs::Vector3Stamped output;
    output.header.stamp = now;
    output.header.frame_id = world_frame_;
    output_pub_.publish(output);
    publishDiagnostics(now, diagnostic_msgs::DiagnosticStatus::WARN, reason, nullptr);
  }

  void tick(const ros::TimerEvent&) {
    const ros::Time now = ros::Time::now();
    if (now.isZero()) return;
    if (!have_policy_ || !fresh(now, policy_.header.stamp, policy_timeout_)) {
      publishZero(now, "policy unavailable or stale");
      return;
    }
    if (!have_odometry_ || !fresh(now, odometry_.header.stamp, odometry_timeout_)) {
      publishZero(now, "odometry unavailable or stale");
      return;
    }
    if (!have_scan_ || !fresh(now, scan_stamp_, scan_timeout_)) {
      publishZero(now, "scan unavailable or stale");
      return;
    }

    const double cosine = std::cos(yaw_);
    const double sine = std::sin(yaw_);
    const dc::Vector3 preferred_body{
        cosine*policy_.vector.x+sine*policy_.vector.y,
        -sine*policy_.vector.x+cosine*policy_.vector.y,
        policy_.vector.z};
    const auto& velocity = odometry_.twist.twist.linear;
    const dc::Vector3 actual_body{velocity.x, velocity.y, velocity.z};
    dc::Vector3 goal_body;
    if (have_goal_) {
      const auto& position = odometry_.pose.pose.position;
      const double goal_world_x = goal_.pose.position.x-position.x;
      const double goal_world_y = goal_.pose.position.y-position.y;
      goal_body = {cosine*goal_world_x+sine*goal_world_y,
                   -sine*goal_world_x+cosine*goal_world_y,
                   goal_.pose.position.z-position.z};
    }
    dynamic_obstacles_ = dynamic_tracker_->obstacles(now.toSec());
    std::vector<dc::ObstaclePoint> combined_obstacles = obstacles_;
    if (have_perception_ && fresh(now, perception_stamp_, perception_timeout_)) {
      combined_obstacles.insert(combined_obstacles.end(), dynamic_obstacles_.begin(),
                                dynamic_obstacles_.end());
    }
    const auto filtered = filter_->filter(
        preferred_body, actual_body, combined_obstacles, goal_body);

    geometry_msgs::Vector3Stamped output;
    output.header.stamp = now;
    output.header.frame_id = world_frame_;
    output.vector.x = cosine*filtered.safe_velocity_body.x-sine*filtered.safe_velocity_body.y;
    output.vector.y = sine*filtered.safe_velocity_body.x+cosine*filtered.safe_velocity_body.y;
    output.vector.z = filtered.safe_velocity_body.z;
    output_pub_.publish(output);
    if (filtered.active) ++intervention_count_;
    publishDiagnostics(now, diagnostic_msgs::DiagnosticStatus::OK,
                       filtered.active ? "intervening" : "pass-through", &filtered);
  }

  void publishDiagnostics(const ros::Time& now, unsigned char level,
                          const std::string& message,
                          const dc::ObstacleFilterOutput* output) {
    if (!last_diagnostic_.isZero() && (now-last_diagnostic_).toSec() < 0.2) return;
    last_diagnostic_ = now;
    diagnostic_msgs::DiagnosticArray array;
    array.header.stamp = now;
    diagnostic_msgs::DiagnosticStatus status;
    status.name = "dnav_obstacle_safety_filter";
    status.hardware_id = "velocity_filter";
    status.level = level;
    status.message = message;
    status.values = {
        keyValue("obstacle_count", asString(obstacles_.size())),
        keyValue("intervention_count", asString(intervention_count_)),
        keyValue("invalid_input_count", asString(invalid_input_count_)),
        keyValue("active_constraints", asString(output ? output->active_constraints : 0)),
        keyValue("dynamic_obstacle_count", asString(dynamic_obstacles_.size())),
        keyValue("dynamic_constraints", asString(output ? output->dynamic_constraints : 0)),
        keyValue("escape_assist", asString(output ? output->escape_assist : false)),
        keyValue("minimum_surface_clearance",
                 asString(output ? output->minimum_surface_clearance : -1.0)),
        keyValue("minimum_allowed_closing_speed",
                 asString(output ? output->minimum_allowed_closing_speed : -1.0)),
    };
    for (std::size_t index = 0; index < dynamic_obstacles_.size() && index < 3;
         ++index) {
      const auto& obstacle = dynamic_obstacles_[index];
      status.values.push_back(keyValue(
          "dynamic_" + std::to_string(index),
          asString(obstacle.x) + "," + asString(obstacle.y) + "," +
              asString(obstacle.relative_radial_velocity) + "," +
              asString(obstacle.relative_tangential_velocity)));
    }
    array.status.push_back(status);
    diagnostics_pub_.publish(array);
  }

  ros::NodeHandle nh_, pnh_;
  std::unique_ptr<dc::ObstacleSafetyFilter> filter_;
  std::unique_ptr<dc::DynamicObstacleTracker> dynamic_tracker_;
  ros::Publisher output_pub_, diagnostics_pub_;
  ros::Subscriber policy_sub_, goal_sub_, odometry_sub_, scan_sub_, perception_sub_;
  ros::Timer timer_;
  geometry_msgs::Vector3Stamped policy_;
  geometry_msgs::PoseStamped goal_;
  nav_msgs::Odometry odometry_;
  std::vector<dc::ObstaclePoint> obstacles_;
  std::vector<dc::ObstaclePoint> dynamic_obstacles_;
  dc::DynamicClusterConfig dynamic_cluster_config_;
  ros::Time scan_stamp_, perception_stamp_, last_diagnostic_;
  bool have_policy_ = false, have_goal_ = false, have_odometry_ = false;
  bool have_scan_ = false, have_perception_ = false;
  double yaw_ = 0.0, rate_, policy_timeout_, odometry_timeout_, scan_timeout_;
  double perception_timeout_, dynamic_max_range_, dynamic_velocity_scale_;
  double dynamic_uncertainty_scale_, dynamic_valid_depth_threshold_;
  double future_tolerance_, vertical_min_, vertical_max_;
  int scan_height_, scan_width_;
  std::string world_frame_, body_frame_, sensor_frame_;
  unsigned long intervention_count_ = 0, invalid_input_count_ = 0;
};

int main(int argc, char** argv) {
  ros::init(argc, argv, "dnav_obstacle_safety_filter");
  try {
    ObstacleSafetyFilterNode node;
    ros::spin();
  } catch (const std::exception& error) {
    ROS_FATAL("Obstacle safety filter failed: %s", error.what());
    return 1;
  }
  return 0;
}
