#include <algorithm>
#include <cmath>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>

#include <diagnostic_msgs/DiagnosticArray.h>
#include <diagnostic_msgs/DiagnosticStatus.h>
#include <diagnostic_msgs/KeyValue.h>
#include <geometry_msgs/TwistStamped.h>
#include <geometry_msgs/Vector3Stamped.h>
#include <nav_msgs/Odometry.h>
#include <ros/ros.h>
#include <std_msgs/Bool.h>
#include <std_msgs/String.h>
#include <std_msgs/UInt8.h>
#include <std_srvs/Trigger.h>

#include <dnav_controller/safety_controller.h>

namespace dc = dnav_controller;

namespace {
constexpr std::uint8_t kFlyingState = 1;
bool isFinite(double value) { return std::isfinite(value); }

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

std::string vectorString(const dc::Vector3& value) {
  std::ostringstream stream;
  stream << value.x << "," << value.y << "," << value.z;
  return stream.str();
}
}  // namespace

class VelocitySafetyAdapter {
 public:
  VelocitySafetyAdapter()
      : nh_(), pnh_("~"), controller_(loadSafetyConfig()),
        control_rate_(parameter("control_rate", 50.0)),
        policy_timeout_(parameter("policy_timeout", 0.10)),
        odometry_timeout_(parameter("odometry_timeout", 0.10)),
        flight_state_timeout_(parameter("flight_state_timeout", 0.20)),
        future_stamp_tolerance_(parameter("future_stamp_tolerance", 0.05)),
        max_control_period_(parameter("max_control_period", 0.10)),
        max_tilt_rad_(parameter("max_tilt_deg", 45.0) * M_PI / 180.0),
        diagnostics_period_(1.0 / parameter("diagnostics_rate", 5.0)),
        world_frame_(parameter<std::string>("world_frame", "map")),
        body_frame_(parameter<std::string>("body_frame", "base_link")),
        output_frame_(parameter<std::string>("output_frame", "base_link")) {
    validateRuntimeConfig();
    command_publisher_ = nh_.advertise<geometry_msgs::TwistStamped>(
        parameter<std::string>("output_topic", "/dnav/uav/cmd_vel"), 1);
    state_publisher_ = nh_.advertise<std_msgs::String>(
        parameter<std::string>("state_topic", "/dnav/controller/state"), 1, true);
    diagnostic_publisher_ = nh_.advertise<diagnostic_msgs::DiagnosticArray>(
        parameter<std::string>("diagnostics_topic", "/dnav/controller/diagnostics"), 1);
    policy_subscriber_ = nh_.subscribe(
        parameter<std::string>("policy_topic", "/dnav/policy/world_velocity"),
        1, &VelocitySafetyAdapter::policy, this);
    odometry_subscriber_ = nh_.subscribe(
        parameter<std::string>("odometry_topic", "/dnav/uav/odom_raw"),
        20, &VelocitySafetyAdapter::odometry, this);
    flight_state_subscriber_ = nh_.subscribe(
        parameter<std::string>("flight_state_topic", "/dnav/uav/flight_state"),
        10, &VelocitySafetyAdapter::flightState, this);
    enable_subscriber_ = nh_.subscribe(
        parameter<std::string>("enable_topic", "/dnav/controller/enable"),
        1, &VelocitySafetyAdapter::enable, this);
    estop_subscriber_ = nh_.subscribe(
        parameter<std::string>("emergency_stop_topic", "/dnav/controller/emergency_stop"),
        1, &VelocitySafetyAdapter::emergencyStop, this);
    reset_estop_service_ = nh_.advertiseService(
        parameter<std::string>("reset_emergency_stop_service",
                               "/dnav/controller/reset_emergency_stop"),
        &VelocitySafetyAdapter::resetEmergencyStop, this);
    timer_ = nh_.createTimer(ros::Duration(1.0 / control_rate_),
                             &VelocitySafetyAdapter::tick, this);
    publishState(dc::ControllerState::DISABLED, "initializing");
    ROS_INFO("dnav velocity safety adapter ready at %.1f Hz", control_rate_);
  }

 private:
  template <typename T>
  T parameter(const std::string& name, const T& default_value) {
    T value;
    pnh_.param<T>(name, value, default_value);
    return value;
  }

  dc::SafetyConfig loadSafetyConfig() {
    dc::SafetyConfig config;
    config.max_horizontal_speed = parameter("max_horizontal_speed", 0.5);
    config.max_ascent_speed = parameter("max_ascent_speed", 0.4);
    config.max_descent_speed = parameter("max_descent_speed", 0.3);
    config.max_horizontal_acceleration = parameter("max_horizontal_acceleration", 0.8);
    config.max_vertical_acceleration = parameter("max_vertical_acceleration", 0.5);
    config.min_altitude = parameter("min_altitude", 0.5);
    config.max_altitude = parameter("max_altitude", 4.0);
    config.altitude_soft_margin = parameter("altitude_soft_margin", 0.5);
    return config;
  }

  void validateRuntimeConfig() {
    if (!(control_rate_ > 0.0) || !(policy_timeout_ > 0.0) ||
        !(odometry_timeout_ > 0.0) || !(flight_state_timeout_ > 0.0) ||
        !(future_stamp_tolerance_ >= 0.0) || !(max_control_period_ > 0.0) ||
        !(max_tilt_rad_ > 0.0 && max_tilt_rad_ < M_PI) ||
        !(diagnostics_period_ > 0.0)) {
      throw std::invalid_argument("invalid dnav_controller runtime configuration");
    }
  }

  void policy(const geometry_msgs::Vector3Stamped::ConstPtr& message) {
    if (message->header.frame_id != world_frame_ || message->header.stamp.isZero() ||
        !isFinite(message->vector.x) || !isFinite(message->vector.y) ||
        !isFinite(message->vector.z)) {
      ++invalid_input_count_;
      ROS_ERROR_THROTTLE(1.0, "Rejected invalid policy velocity");
      return;
    }
    policy_velocity_world_ = {message->vector.x, message->vector.y, message->vector.z};
    policy_stamp_ = message->header.stamp;
    policy_receive_time_ = ros::Time::now();
    have_policy_ = true;
  }

  void odometry(const nav_msgs::Odometry::ConstPtr& message) {
    const auto& position = message->pose.pose.position;
    const auto& q = message->pose.pose.orientation;
    const double norm = std::sqrt(q.w*q.w + q.x*q.x + q.y*q.y + q.z*q.z);
    if (message->header.frame_id != world_frame_ ||
        message->child_frame_id != body_frame_ || message->header.stamp.isZero() ||
        !isFinite(position.z) || !isFinite(norm) || norm < 1e-6) {
      ++invalid_input_count_;
      ROS_ERROR_THROTTLE(1.0, "Rejected invalid odometry");
      return;
    }
    const double w=q.w/norm, x=q.x/norm, y=q.y/norm, z=q.z/norm;
    const double roll = std::atan2(2.0*(w*x+y*z), 1.0-2.0*(x*x+y*y));
    const double pitch = std::asin(std::max(-1.0, std::min(1.0, 2.0*(w*y-z*x))));
    yaw_ = std::atan2(2.0*(w*z+x*y), 1.0-2.0*(y*y+z*z));
    altitude_ = position.z;
    attitude_valid_ = isFinite(roll) && isFinite(pitch) && isFinite(yaw_) &&
                      std::hypot(roll, pitch) <= max_tilt_rad_;
    odometry_stamp_ = message->header.stamp;
    odometry_receive_time_ = ros::Time::now();
    have_odometry_ = true;
  }

  void flightState(const std_msgs::UInt8::ConstPtr& message) {
    if (message->data > 3) {
      ++invalid_input_count_;
      ROS_ERROR_THROTTLE(1.0, "Rejected unknown flight state: %u", message->data);
      return;
    }
    flight_state_ = message->data;
    flight_state_receive_time_ = ros::Time::now();
    have_flight_state_ = true;
  }

  void enable(const std_msgs::Bool::ConstPtr& message) {
    navigation_enabled_ = message->data;
    if (!navigation_enabled_) controller_.resetLimiter();
  }

  void emergencyStop(const std_msgs::Bool::ConstPtr& message) {
    if (message->data && !estop_latched_) {
      estop_latched_ = true;
      ++estop_count_;
      controller_.resetLimiter();
      ROS_ERROR("Emergency stop latched; publishing zero velocity");
    }
  }

  bool resetEmergencyStop(std_srvs::Trigger::Request&,
                          std_srvs::Trigger::Response& response) {
    const ros::Time now = ros::Time::now();
    const bool odometry_fresh = have_odometry_ &&
        fresh(now, odometry_stamp_, odometry_receive_time_, odometry_timeout_);
    if (navigation_enabled_) {
      response.success = false;
      response.message = "disable navigation before resetting emergency stop";
    } else if (!odometry_fresh || !attitude_valid_) {
      response.success = false;
      response.message = "valid fresh odometry is required";
    } else {
      estop_latched_ = false;
      controller_.resetLimiter();
      response.success = true;
      response.message = "emergency stop reset; navigation remains disabled";
    }
    return true;
  }

  bool fresh(const ros::Time& now, const ros::Time& stamp,
             const ros::Time& received, double timeout) const {
    const double stamp_age = (now-stamp).toSec();
    const double receive_age = (now-received).toSec();
    return stamp_age >= -future_stamp_tolerance_ && stamp_age <= timeout &&
           receive_age >= -future_stamp_tolerance_ && receive_age <= timeout;
  }

  void tick(const ros::TimerEvent&) {
    const ros::Time now = ros::Time::now();
    if (!last_clock_.isZero() && now < last_clock_) {
      have_policy_ = have_odometry_ = have_flight_state_ = false;
      navigation_enabled_ = false;
      controller_.resetLimiter();
      // All scheduling baselines derived from ROS time must be discarded.
      // Otherwise a simulation reset can suppress diagnostics until the new
      // clock catches up with the pre-reset timestamp.
      last_diagnostic_time_ = ros::Time();
      ++time_rollback_count_;
    }
    double dt = 1.0/control_rate_;
    if (!last_control_time_.isZero()) dt = (now-last_control_time_).toSec();
    last_clock_ = now;
    last_control_time_ = now;
    const bool odometry_valid = have_odometry_ &&
        fresh(now, odometry_stamp_, odometry_receive_time_, odometry_timeout_);
    const bool policy_valid = have_policy_ &&
        fresh(now, policy_stamp_, policy_receive_time_, policy_timeout_);
    const double flight_age = (now-flight_state_receive_time_).toSec();
    const bool flight_valid = have_flight_state_ &&
        flight_age >= -future_stamp_tolerance_ && flight_age <= flight_state_timeout_;

    dc::ControllerInput input;
    input.navigation_enabled = navigation_enabled_;
    input.estop_latched = estop_latched_;
    input.odometry_valid = odometry_valid;
    input.attitude_valid = attitude_valid_;
    input.flight_state_valid = flight_valid;
    input.flight_state_flying = flight_state_ == kFlyingState;
    input.policy_valid = policy_valid;
    input.policy_velocity_world = policy_velocity_world_;
    input.altitude = altitude_;
    input.yaw = yaw_;
    input.dt = dt <= max_control_period_ ? dt : 0.0;
    const dc::ControllerOutput output = controller_.update(input);

    geometry_msgs::TwistStamped command;
    command.header.stamp = now;
    command.header.frame_id = output_frame_;
    command.twist.linear.x = output.command_velocity_heading.x;
    command.twist.linear.y = output.command_velocity_heading.y;
    command.twist.linear.z = output.command_velocity_heading.z;
    command_publisher_.publish(command);
    ++published_command_count_;
    if (output.horizontal_speed_limited) ++horizontal_limit_count_;
    if (output.vertical_speed_limited) ++vertical_limit_count_;
    if (output.acceleration_limited) ++acceleration_limit_count_;
    if (output.altitude_limited) ++altitude_limit_count_;
    publishState(output.state, output.reason);
    if (last_diagnostic_time_.isZero() ||
        (now-last_diagnostic_time_).toSec() >= diagnostics_period_) {
      publishDiagnostics(now, output, policy_valid, odometry_valid, flight_valid);
      last_diagnostic_time_ = now;
    }
  }

  void publishState(dc::ControllerState state, const std::string& reason) {
    if (state == last_state_ && reason == last_reason_) return;
    last_state_ = state;
    last_reason_ = reason;
    std_msgs::String message;
    message.data = std::string(dc::stateName(state)) + ": " + reason;
    state_publisher_.publish(message);
  }

  void publishDiagnostics(const ros::Time& now, const dc::ControllerOutput& output,
                          bool policy_valid, bool odometry_valid, bool flight_valid) {
    diagnostic_msgs::DiagnosticArray array;
    array.header.stamp = now;
    diagnostic_msgs::DiagnosticStatus status;
    status.name = "dnav_controller";
    status.hardware_id = "velocity_safety_adapter";
    status.message = output.reason;
    status.level = output.state == dc::ControllerState::ACTIVE
        ? diagnostic_msgs::DiagnosticStatus::OK
        : ((output.state == dc::ControllerState::FAULT ||
            output.state == dc::ControllerState::ESTOP)
           ? diagnostic_msgs::DiagnosticStatus::ERROR
           : diagnostic_msgs::DiagnosticStatus::WARN);
    status.values = {
      keyValue("controller_state", dc::stateName(output.state)),
      keyValue("navigation_enabled", navigation_enabled_ ? "true" : "false"),
      keyValue("estop_latched", estop_latched_ ? "true" : "false"),
      keyValue("flight_state", asString(static_cast<int>(flight_state_))),
      keyValue("policy_valid", policy_valid ? "true" : "false"),
      keyValue("odometry_valid", odometry_valid ? "true" : "false"),
      keyValue("flight_state_valid", flight_valid ? "true" : "false"),
      keyValue("raw_world_velocity", vectorString(output.raw_velocity_world)),
      keyValue("limited_world_velocity", vectorString(output.limited_velocity_world)),
      keyValue("output_heading_velocity", vectorString(output.command_velocity_heading)),
      keyValue("invalid_input_count", asString(invalid_input_count_)),
      keyValue("time_rollback_count", asString(time_rollback_count_)),
      keyValue("estop_count", asString(estop_count_)),
      keyValue("horizontal_limit_count", asString(horizontal_limit_count_)),
      keyValue("vertical_limit_count", asString(vertical_limit_count_)),
      keyValue("acceleration_limit_count", asString(acceleration_limit_count_)),
      keyValue("altitude_limit_count", asString(altitude_limit_count_)),
      keyValue("published_command_count", asString(published_command_count_))
    };
    array.status.push_back(status);
    diagnostic_publisher_.publish(array);
  }

  ros::NodeHandle nh_, pnh_;
  dc::SafetyController controller_;
  double control_rate_, policy_timeout_, odometry_timeout_, flight_state_timeout_;
  double future_stamp_tolerance_, max_control_period_, max_tilt_rad_, diagnostics_period_;
  std::string world_frame_, body_frame_, output_frame_;
  ros::Publisher command_publisher_, state_publisher_, diagnostic_publisher_;
  ros::Subscriber policy_subscriber_, odometry_subscriber_, flight_state_subscriber_;
  ros::Subscriber enable_subscriber_, estop_subscriber_;
  ros::ServiceServer reset_estop_service_;
  ros::Timer timer_;
  dc::Vector3 policy_velocity_world_;
  double altitude_=0.0, yaw_=0.0;
  bool attitude_valid_=false;
  std::uint8_t flight_state_=0;
  bool have_policy_=false, have_odometry_=false, have_flight_state_=false;
  bool navigation_enabled_=false, estop_latched_=false;
  ros::Time policy_stamp_, policy_receive_time_, odometry_stamp_, odometry_receive_time_;
  ros::Time flight_state_receive_time_, last_clock_, last_control_time_, last_diagnostic_time_;
  dc::ControllerState last_state_=dc::ControllerState::FAULT;
  std::string last_reason_;
  std::uint64_t invalid_input_count_=0, time_rollback_count_=0, estop_count_=0;
  std::uint64_t horizontal_limit_count_=0, vertical_limit_count_=0;
  std::uint64_t acceleration_limit_count_=0, altitude_limit_count_=0;
  std::uint64_t published_command_count_=0;
};

int main(int argc, char** argv) {
  ros::init(argc, argv, "dnav_velocity_controller");
  try {
    VelocitySafetyAdapter adapter;
    ros::spin();
  } catch (const std::exception& error) {
    ROS_FATAL("Failed to start dnav controller: %s", error.what());
    return 1;
  }
  return 0;
}
