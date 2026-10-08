#!/usr/bin/env python3
"""Run one configured Safe-DNav evaluation episode."""

import math
from pathlib import Path
import signal
import subprocess
from threading import Lock

import numpy as np
import rospkg
import rospy
from gazebo_msgs.msg import ContactsState
from geometry_msgs.msg import PoseStamped, TwistStamped, Vector3Stamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, Empty, String, UInt8

from dnav_evaluation.contact import ContactKind, classify_vehicle_contact
from dnav_evaluation.episode import EpisodeMonitor, EpisodeOutcome
from dnav_evaluation.reporting import write_episode_result
from dnav_evaluation.reporting import result_basename
from dnav_evaluation.reporting import write_trajectory
from dnav_evaluation.scenario import (
    load_scenarios,
    validate_spawn_configuration,
    validate_world_file,
)
from dnav_perception.msg import PerceptionObservation


FLYING = 1


class EpisodeRunner:
    def __init__(self):
        package = Path(rospkg.RosPack().get_path("dnav_evaluation"))
        scenario_file = Path(rospy.get_param("~scenario_file", str(package/"config/empty_smoke.yaml")))
        requested_name = rospy.get_param("~scenario_name", "empty_forward_yaw0")
        matches = [item for item in load_scenarios(str(scenario_file)) if item.name == requested_name]
        if len(matches) != 1:
            raise ValueError(f"scenario not found exactly once: {requested_name}")
        self.scenario = matches[0]
        configured_world = rospy.get_param("~world_file")
        gazebo_package = rospkg.RosPack().get_path("dnav_gazebo")
        self.world_file = validate_world_file(
            configured_world, self.scenario.world, gazebo_package
        )
        validate_spawn_configuration(
            (
                float(rospy.get_param("~spawn_x")),
                float(rospy.get_param("~spawn_y")),
                float(rospy.get_param("~spawn_z")),
            ),
            float(rospy.get_param("~spawn_yaw")),
            self.scenario,
        )
        self.output_directory = rospy.get_param("~output_directory", "/tmp/dnav_evaluation")
        self.run_id = str(rospy.get_param("~run_id", ""))
        self.settle_time = float(rospy.get_param("~settle_time", 1.0))
        self.max_range = float(rospy.get_param("~perception_max_range", 10.0))
        self.record_bag = bool(rospy.get_param("~record_bag", False))
        self.monitor = EpisodeMonitor(self.scenario.episode)
        self.lock = Lock()
        self.latest_odometry = None
        self.latest_raw_clearance = None
        self.latest_scan_stamp = None
        self.latest_scan_receive_time = None
        self.latest_static_clearance = None
        self.latest_dynamic_clearance = None
        self.latest_perception_stamp = None
        self.latest_perception_receive_time = None
        self.latest_history_valid = False
        self.latest_policy = None
        self.latest_safe_velocity = None
        self.latest_command = None
        self.flight_state = 0
        self.flying_since = None
        self.controller_state = ""
        self.active = False
        self.done = False
        self.trajectory = []
        self.last_trajectory_time = None
        self.bag_process = None
        self.bag_path = None

        self.takeoff_publisher = rospy.Publisher("/dnav/uav/takeoff", Empty, queue_size=1)
        self.goal_publisher = rospy.Publisher("/dnav/policy/goal", PoseStamped, queue_size=1, latch=True)
        self.enable_publisher = rospy.Publisher("/dnav/controller/enable", Bool, queue_size=1, latch=True)
        self.estop_publisher = rospy.Publisher(
            "/dnav/controller/emergency_stop", Bool, queue_size=1, latch=True
        )
        self.result_publisher = rospy.Publisher(
            "/dnav/evaluation/result", String, queue_size=1, latch=True
        )
        rospy.Subscriber("/dnav/uav/odom_raw", Odometry, self._odometry, queue_size=50)
        rospy.Subscriber("/dnav/uav/flight_state", UInt8, self._flight_state, queue_size=10)
        rospy.Subscriber("/dnav/uav/contacts", ContactsState, self._contacts, queue_size=10)
        rospy.Subscriber("/dnav/uav/mid360/scan", LaserScan, self._scan, queue_size=1)
        rospy.Subscriber(
            "/dnav/perception/observation", PerceptionObservation, self._perception, queue_size=1
        )
        rospy.Subscriber("/dnav/controller/state", String, self._controller_state, queue_size=10)
        rospy.Subscriber(
            "/dnav/policy/world_velocity", Vector3Stamped, self._policy_command, queue_size=10
        )
        rospy.Subscriber(
            "/dnav/safety/world_velocity", Vector3Stamped, self._safe_velocity, queue_size=10
        )
        rospy.Subscriber("/dnav/uav/cmd_vel", TwistStamped, self._controller_command, queue_size=10)
        self.timer = rospy.Timer(rospy.Duration(0.02), self._tick)
        rospy.on_shutdown(self._safe_stop)
        if self.record_bag:
            self._start_bag()
        rospy.loginfo("Evaluation scenario loaded: %s", self.scenario.name)

    def _odometry(self, message):
        with self.lock:
            self.latest_odometry = message

    def _flight_state(self, message):
        with self.lock:
            self.flight_state = message.data

    def _controller_state(self, message):
        with self.lock:
            self.controller_state = message.data

    def _policy_command(self, message):
        with self.lock:
            self.latest_policy = message

    def _controller_command(self, message):
        with self.lock:
            self.latest_command = message

    def _safe_velocity(self, message):
        with self.lock:
            self.latest_safe_velocity = message

    def _scan(self, message):
        finite = [value for value in message.ranges if math.isfinite(value) and value >= message.range_min]
        with self.lock:
            self.latest_raw_clearance = min(finite) if finite else None
            self.latest_scan_stamp = message.header.stamp.to_sec()
            self.latest_scan_receive_time = rospy.Time.now().to_sec()

    def _perception(self, message):
        static = np.asarray(message.lidar_static, dtype=np.float32)
        dynamic = np.asarray(message.lidar_dynamic, dtype=np.float32)
        if static.size != 216 or dynamic.size != 1080:
            return
        static_clearance = float(np.min(static))*self.max_range
        dynamic_clearance = float(np.min(dynamic.reshape(5, 6, 36)[0]))*self.max_range
        with self.lock:
            self.latest_static_clearance = static_clearance
            self.latest_dynamic_clearance = dynamic_clearance
            self.latest_perception_stamp = message.header.stamp.to_sec()
            self.latest_perception_receive_time = rospy.Time.now().to_sec()
            self.latest_history_valid = bool(message.history_valid)

    def _contacts(self, message):
        if not self.active or self.done:
            return
        for state in message.states:
            kind = classify_vehicle_contact(
                state.collision1_name,
                state.collision2_name,
                dynamic_tokens=self.scenario.dynamic_model_tokens,
            )
            if kind is None:
                continue
            detail = f"{state.collision1_name} <-> {state.collision2_name}"
            with self.lock:
                outcome = self.monitor.collision(kind == ContactKind.DYNAMIC, detail)
            self._finish(outcome)
            return

    def _publish_goal_and_enable(self, now):
        goal = PoseStamped()
        goal.header.stamp = now
        goal.header.frame_id = "map"
        goal.pose.position.x, goal.pose.position.y, goal.pose.position.z = self.scenario.episode.goal
        goal.pose.orientation.w = 1.0
        self.goal_publisher.publish(goal)
        self.estop_publisher.publish(Bool(data=False))
        self.enable_publisher.publish(Bool(data=True))

    def _tick(self, _event):
        if self.done:
            return
        now = rospy.Time.now()
        with self.lock:
            odometry = self.latest_odometry
            flight_state = self.flight_state
            raw = self.latest_raw_clearance
            static = self.latest_static_clearance
            dynamic = self.latest_dynamic_clearance
            scan_stamp = self.latest_scan_stamp
            scan_receive_time = self.latest_scan_receive_time
            perception_stamp = self.latest_perception_stamp
            perception_receive_time = self.latest_perception_receive_time
            history_valid = self.latest_history_valid
            policy = self.latest_policy
            safe_velocity = self.latest_safe_velocity
            command = self.latest_command
            controller_state = self.controller_state
        if odometry is None or now.is_zero():
            return
        if flight_state != FLYING:
            self.flying_since = None
            self.takeoff_publisher.publish(Empty())
            return
        if self.flying_since is None:
            self.flying_since = now
            return
        if not self.active:
            if (now-self.flying_since).to_sec() < self.settle_time:
                return
            position = odometry.pose.pose.position
            self.monitor.start(now.to_sec(), (position.x, position.y, position.z))
            self._publish_goal_and_enable(now)
            self.active = True
            rospy.loginfo("Evaluation episode started: %s", self.scenario.name)
            return

        position = odometry.pose.pose.position
        velocity = odometry.twist.twist.linear
        if self.last_trajectory_time is None or (now.to_sec()-self.last_trajectory_time) >= 0.1:
            self.trajectory.append({
                "time": now.to_sec()-self.monitor.start_time,
                "x": position.x,
                "y": position.y,
                "z": position.z,
                "vx": velocity.x,
                "vy": velocity.y,
                "vz": velocity.z,
                "goal_distance": math.sqrt(
                    (position.x-self.scenario.episode.goal[0])**2
                    +(position.y-self.scenario.episode.goal[1])**2
                    +(position.z-self.scenario.episode.goal[2])**2
                ),
                "raw_clearance": raw,
                "static_clearance": static,
                "dynamic_clearance": dynamic,
                "odom_stamp": odometry.header.stamp.to_sec(),
                "odom_age": now.to_sec()-odometry.header.stamp.to_sec(),
                "scan_stamp": scan_stamp,
                "scan_age": None if scan_stamp is None else now.to_sec()-scan_stamp,
                "scan_transport_age": None if scan_stamp is None or scan_receive_time is None
                    else scan_receive_time-scan_stamp,
                "perception_stamp": perception_stamp,
                "perception_age": None if perception_stamp is None else now.to_sec()-perception_stamp,
                "perception_pipeline_age": None
                    if perception_stamp is None or perception_receive_time is None
                    else perception_receive_time-perception_stamp,
                "history_valid": history_valid,
                "policy_stamp": None if policy is None else policy.header.stamp.to_sec(),
                "policy_age": None if policy is None else now.to_sec()-policy.header.stamp.to_sec(),
                "policy_vx": None if policy is None else policy.vector.x,
                "policy_vy": None if policy is None else policy.vector.y,
                "policy_vz": None if policy is None else policy.vector.z,
                "safety_stamp": None if safe_velocity is None else safe_velocity.header.stamp.to_sec(),
                "safety_age": None if safe_velocity is None
                    else now.to_sec()-safe_velocity.header.stamp.to_sec(),
                "safety_vx": None if safe_velocity is None else safe_velocity.vector.x,
                "safety_vy": None if safe_velocity is None else safe_velocity.vector.y,
                "safety_vz": None if safe_velocity is None else safe_velocity.vector.z,
                "command_stamp": None if command is None else command.header.stamp.to_sec(),
                "command_age": None if command is None else now.to_sec()-command.header.stamp.to_sec(),
                "command_vx": None if command is None else command.twist.linear.x,
                "command_vy": None if command is None else command.twist.linear.y,
                "command_vz": None if command is None else command.twist.linear.z,
                "controller_state": controller_state,
            })
            self.last_trajectory_time = now.to_sec()
        system_fault = None
        if controller_state.startswith("FAULT:") or controller_state.startswith("ESTOP:"):
            system_fault = controller_state
        with self.lock:
            outcome = self.monitor.update(
                now.to_sec(),
                (position.x, position.y, position.z),
                (velocity.x, velocity.y, velocity.z),
                raw_clearance=raw,
                static_clearance=static,
                dynamic_clearance=dynamic,
                system_fault=system_fault,
            )
        if outcome != EpisodeOutcome.RUNNING:
            self._finish(outcome)

    def _finish(self, outcome):
        with self.lock:
            if self.done:
                return
            self.done = True
            result = self.monitor.result().to_dict()
        self._safe_stop()
        self._stop_bag()
        result.update({
            "world": self.scenario.world,
            "world_file": str(self.world_file),
            "configured_start": list(self.scenario.start),
            "goal": list(self.scenario.episode.goal),
            "yaw": self.scenario.yaw,
            "max_horizontal_speed": self.scenario.max_horizontal_speed,
            "bag_file": self.bag_path,
        })
        trajectory = write_trajectory(
            self.output_directory, self.scenario.name, self.trajectory, run_id=self.run_id
        )
        result["trajectory_file"] = str(trajectory)
        destination = write_episode_result(
            self.output_directory, self.scenario.name, result, run_id=self.run_id
        )
        self.result_publisher.publish(String(data=str(destination)))
        rospy.loginfo("Evaluation finished: %s (%s), result=%s", self.scenario.name, outcome.value, destination)
        rospy.signal_shutdown(f"episode finished: {outcome.value}")

    def _safe_stop(self):
        self.enable_publisher.publish(Bool(data=False))
        if self.active:
            self.estop_publisher.publish(Bool(data=True))
        self._stop_bag()

    def _start_bag(self):
        directory = Path(self.output_directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.bag_path = str(
            directory/f"{result_basename(self.scenario.name, self.run_id)}.bag"
        )
        topics = [
            "/clock", "/tf", "/tf_static", "/dnav/uav/odom_raw",
            "/dnav/uav/mid360/scan", "/dnav/perception/observation",
            "/dnav/policy/world_velocity", "/dnav/uav/cmd_vel",
            "/dnav/safety/world_velocity", "/dnav/safety/diagnostics",
            "/dnav/controller/state", "/dnav/controller/diagnostics",
            "/dnav/uav/contacts", "/dnav/uav/flight_state",
        ]
        self.bag_process = subprocess.Popen(
            ["rosbag", "record", "--buffsize=128", "-O", self.bag_path, *topics],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def _stop_bag(self):
        if self.bag_process is None or self.bag_process.poll() is not None:
            return
        self.bag_process.send_signal(signal.SIGINT)
        try:
            self.bag_process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            self.bag_process.terminate()
            self.bag_process.wait(timeout=2.0)


def main():
    rospy.init_node("dnav_episode_runner")
    try:
        EpisodeRunner()
        rospy.spin()
    except Exception as error:
        rospy.logfatal("Evaluation runner failed: %s", error)
        raise


if __name__ == "__main__":
    main()
