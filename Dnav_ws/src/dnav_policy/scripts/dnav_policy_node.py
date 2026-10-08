#!/usr/bin/env python3
"""ROS1 node for deterministic Safe-DNav policy inference."""

from pathlib import Path
import time

import numpy as np
import rospkg
import rospy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import PoseStamped, Vector3Stamped
from nav_msgs.msg import Odometry

from dnav_perception.msg import PerceptionObservation
from dnav_policy.geometry import ros_xyzw_to_wxyz
from dnav_policy.runtime import PolicyRuntime
from dnav_policy.state_builder import PolicyStateBuilder
from dnav_policy.synchronization import (
    GoalSample,
    InputUnavailable,
    OdometrySample,
    PerceptionSample,
    PolicyInputCache,
)


class PolicyNode:
    def __init__(self):
        package_path = Path(rospkg.RosPack().get_path("dnav_policy"))
        model_setting = Path(rospy.get_param("~model_file", "weights/policy_best.pt"))
        if model_setting.is_absolute():
            model_path = model_setting
        else:
            package_candidate = package_path / model_setting
            workspace_candidate = package_path.parents[1] / model_setting
            model_path = package_candidate if package_candidate.is_file() else workspace_candidate

        self.world_frame = rospy.get_param("~world_frame", "map")
        self.body_frame = rospy.get_param("~body_frame", "base_link")
        self.perception_timeout = float(rospy.get_param("~perception_timeout", 0.25))
        self.odometry_timeout = float(rospy.get_param("~odometry_timeout", 0.10))
        policy_rate = float(rospy.get_param("~policy_rate", 62.5))
        if policy_rate <= 0.0 or self.perception_timeout <= 0.0 or self.odometry_timeout <= 0.0:
            raise ValueError("policy rate and input timeouts must be positive")

        self.runtime = PolicyRuntime(
            model_path,
            device=rospy.get_param("~device", "cuda"),
            fallback_to_cpu=bool(rospy.get_param("~fallback_to_cpu", True)),
            action_limit=float(rospy.get_param("~action_limit", 3.0)),
        )
        self.state_builder = PolicyStateBuilder(
            velocity_scale=float(rospy.get_param("~state_velocity_scale", 3.6)),
            angular_velocity_scale=float(
                rospy.get_param("~state_angular_velocity_scale", 2.0 * np.pi)
            ),
            distance_scale=float(
                rospy.get_param("~state_distance_scale", PolicyStateBuilder.STATE_DISTANCE_SCALE)
            ),
        )
        self.cache = PolicyInputCache()
        self.state_generation = -1
        self.last_status = "initializing"
        self.error_count = 0
        self.publish_count = 0

        self.publisher = rospy.Publisher(
            rospy.get_param("~output_topic", "/dnav/policy/world_velocity"),
            Vector3Stamped,
            queue_size=1,
        )
        self.diagnostic_publisher = rospy.Publisher(
            rospy.get_param("~diagnostics_topic", "/dnav/policy/diagnostics"),
            DiagnosticArray,
            queue_size=1,
        )
        rospy.Subscriber(
            rospy.get_param("~perception_topic", "/dnav/perception/observation"),
            PerceptionObservation,
            self._perception_callback,
            queue_size=1,
        )
        rospy.Subscriber(
            rospy.get_param("~odom_topic", "/dnav/uav/odom_raw"),
            Odometry,
            self._odometry_callback,
            queue_size=50,
        )
        rospy.Subscriber(
            rospy.get_param("~goal_topic", "/dnav/policy/goal"),
            PoseStamped,
            self._goal_callback,
            queue_size=1,
        )
        self.timer = rospy.Timer(rospy.Duration(1.0 / policy_rate), self._tick)
        rospy.loginfo(
            "Policy loaded on %s: %s (sha256=%s)",
            self.runtime.device,
            model_path,
            self.runtime.checkpoint_sha256,
        )

    def _perception_callback(self, message):
        try:
            static = np.asarray(message.lidar_static, dtype=np.float32).reshape(1, 6, 36)
            dynamic = np.asarray(message.lidar_dynamic, dtype=np.float32).reshape(5, 6, 36)
            if not np.isfinite(static).all() or not np.isfinite(dynamic).all():
                raise ValueError("perception contains NaN or Inf")
            self.cache.update_perception(
                PerceptionSample(
                    message.header.stamp.to_sec(), static, dynamic, message.history_valid
                )
            )
        except (ValueError, TypeError) as error:
            self.error_count += 1
            rospy.logerr_throttle(1.0, "Rejected perception input: %s", error)

    def _odometry_callback(self, message):
        try:
            if message.header.frame_id and message.header.frame_id != self.world_frame:
                raise ValueError(
                    f"odometry frame must be {self.world_frame}, got {message.header.frame_id}"
                )
            if message.child_frame_id and message.child_frame_id != self.body_frame:
                raise ValueError(
                    f"odometry child frame must be {self.body_frame}, got {message.child_frame_id}"
                )
            pose = message.pose.pose
            twist = message.twist.twist
            quaternion = ros_xyzw_to_wxyz(
                [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
            )
            sample = OdometrySample(
                stamp=message.header.stamp.to_sec(),
                position_world=np.array(
                    [pose.position.x, pose.position.y, pose.position.z], dtype=np.float64
                ),
                quaternion_world_body_wxyz=quaternion,
                linear_velocity_body=np.array(
                    [twist.linear.x, twist.linear.y, twist.linear.z], dtype=np.float64
                ),
                angular_velocity_body=np.array(
                    [twist.angular.x, twist.angular.y, twist.angular.z], dtype=np.float64
                ),
            )
            for value in (
                sample.position_world,
                sample.linear_velocity_body,
                sample.angular_velocity_body,
            ):
                if not np.isfinite(value).all():
                    raise ValueError("odometry contains NaN or Inf")
            self.cache.update_odometry(sample)
        except (ValueError, TypeError) as error:
            self.error_count += 1
            rospy.logerr_throttle(1.0, "Rejected odometry: %s", error)

    def _goal_callback(self, message):
        try:
            frame = message.header.frame_id or self.world_frame
            if frame != self.world_frame:
                raise ValueError(f"goal frame must be {self.world_frame}, got {frame}")
            goal = np.array(
                [message.pose.position.x, message.pose.position.y, message.pose.position.z],
                dtype=np.float64,
            )
            if not np.isfinite(goal).all():
                raise ValueError("goal contains NaN or Inf")
            self.cache.update_goal(GoalSample(message.header.stamp.to_sec(), goal))
        except (ValueError, TypeError) as error:
            self.error_count += 1
            rospy.logerr_throttle(1.0, "Rejected navigation goal: %s", error)

    def _tick(self, _event):
        started = time.perf_counter()
        try:
            now = rospy.Time.now()
            perception, odometry, goal, generation = self.cache.snapshot(
                now.to_sec(),
                perception_timeout=self.perception_timeout,
                odometry_timeout=self.odometry_timeout,
            )
            if generation != self.state_generation:
                self.state_builder.reset()
                self.state_generation = generation
            state = self.state_builder.build(
                odometry.position_world,
                odometry.quaternion_world_body_wxyz,
                odometry.linear_velocity_body,
                odometry.angular_velocity_body,
                goal.position_world,
            )
            output = self.runtime.infer(
                perception.lidar_static,
                perception.lidar_dynamic,
                state,
                odometry.quaternion_world_body_wxyz,
            )
            command = Vector3Stamped()
            command.header.stamp = now
            command.header.frame_id = self.world_frame
            command.vector.x, command.vector.y, command.vector.z = map(
                float, output.velocity_world
            )
            self.publisher.publish(command)
            self.publish_count += 1
            self.last_status = "policy active"
            self._publish_diagnostic(
                DiagnosticStatus.OK,
                self.last_status,
                (time.perf_counter() - started) * 1000.0,
            )
        except InputUnavailable as error:
            self.last_status = str(error)
            self._publish_diagnostic(DiagnosticStatus.WARN, self.last_status, 0.0)
        except Exception as error:
            self.error_count += 1
            self.last_status = str(error)
            rospy.logerr_throttle(1.0, "Policy inference failed: %s", error)
            self._publish_diagnostic(DiagnosticStatus.ERROR, self.last_status, 0.0)

    def _publish_diagnostic(self, level, message, cycle_latency_ms):
        array = DiagnosticArray()
        array.header.stamp = rospy.Time.now()
        status = DiagnosticStatus(
            level=level,
            name="dnav_policy",
            message=message,
            hardware_id=str(self.runtime.device),
        )
        status.values = [
            KeyValue("device", str(self.runtime.device)),
            KeyValue("checkpoint_sha256", self.runtime.checkpoint_sha256),
            KeyValue("inference_latency_ms", f"{self.runtime.last_latency_ms:.3f}"),
            KeyValue("cycle_latency_ms", f"{cycle_latency_ms:.3f}"),
            KeyValue("published_commands", str(self.publish_count)),
            KeyValue("errors", str(self.error_count)),
        ]
        array.status = [status]
        self.diagnostic_publisher.publish(array)


def main():
    rospy.init_node("dnav_policy")
    PolicyNode()
    rospy.spin()


if __name__ == "__main__":
    main()
