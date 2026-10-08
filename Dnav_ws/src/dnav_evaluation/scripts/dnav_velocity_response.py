#!/usr/bin/env python3
"""Measure the deployed controller and Gazebo vehicle velocity response."""

import json
import math
import os
from pathlib import Path
import tempfile
from threading import Lock

from geometry_msgs.msg import TwistStamped, Vector3Stamped
from nav_msgs.msg import Odometry
import rospy
from std_msgs.msg import Bool, Empty, UInt8

from dnav_evaluation.reporting import write_trajectory
from dnav_evaluation.response import compute_response_metrics


FLYING = 1
PHASES = (
    ("warmup", 1.0, (0.0, 0.0, 0.0)),
    ("accelerate", 3.0, (0.5, 0.0, 0.0)),
    ("brake", 3.0, (0.0, 0.0, 0.0)),
    ("pre_turn", 3.0, (0.5, 0.0, 0.0)),
    ("turn", 3.0, (0.0, 0.5, 0.0)),
    ("final_brake", 3.0, (0.0, 0.0, 0.0)),
)


def rotate_vector(q, vector):
    """Rotate a body-frame vector into the world frame with a unit quaternion."""
    norm = math.sqrt(q.w*q.w+q.x*q.x+q.y*q.y+q.z*q.z)
    if norm < 1e-9:
        raise ValueError("invalid odometry quaternion")
    w, x, y, z = q.w/norm, q.x/norm, q.y/norm, q.z/norm
    vx, vy, vz = vector.x, vector.y, vector.z
    return (
        (1-2*(y*y+z*z))*vx + 2*(x*y-z*w)*vy + 2*(x*z+y*w)*vz,
        2*(x*y+z*w)*vx + (1-2*(x*x+z*z))*vy + 2*(y*z-x*w)*vz,
        2*(x*z-y*w)*vx + 2*(y*z+x*w)*vy + (1-2*(x*x+y*y))*vz,
    )


class VelocityResponseRunner:
    def __init__(self):
        self.output_directory = Path(rospy.get_param("~output_directory", "/tmp/dnav_response"))
        self.run_id = str(rospy.get_param("~run_id", "production"))
        self.controller_config = str(rospy.get_param("~controller_config", ""))
        self.settle_time = float(rospy.get_param("~settle_time", 1.0))
        self.lock = Lock()
        self.odometry = None
        self.flight_state = 0
        self.command = None
        self.flying_since = None
        self.start_time = None
        self.samples = []
        self.done = False

        self.takeoff_pub = rospy.Publisher("/dnav/uav/takeoff", Empty, queue_size=1)
        self.enable_pub = rospy.Publisher("/dnav/controller/enable", Bool, queue_size=1, latch=True)
        self.policy_pub = rospy.Publisher(
            "/dnav/policy/world_velocity", Vector3Stamped, queue_size=1
        )
        rospy.Subscriber("/dnav/uav/odom_raw", Odometry, self._odometry, queue_size=20)
        rospy.Subscriber("/dnav/uav/flight_state", UInt8, self._flight_state, queue_size=10)
        rospy.Subscriber("/dnav/uav/cmd_vel", TwistStamped, self._command, queue_size=10)
        self.timer = rospy.Timer(rospy.Duration(0.02), self._tick)
        rospy.on_shutdown(self._stop)

    def _odometry(self, message):
        with self.lock:
            self.odometry = message

    def _flight_state(self, message):
        with self.lock:
            self.flight_state = message.data

    def _command(self, message):
        with self.lock:
            self.command = message

    @staticmethod
    def _phase(elapsed):
        start = 0.0
        for name, duration, velocity in PHASES:
            if elapsed < start+duration:
                return name, elapsed-start, velocity
            start += duration
        return None, None, (0.0, 0.0, 0.0)

    def _publish_policy(self, now, velocity):
        message = Vector3Stamped()
        message.header.stamp = now
        message.header.frame_id = "map"
        message.vector.x, message.vector.y, message.vector.z = velocity
        self.policy_pub.publish(message)

    def _tick(self, _event):
        if self.done:
            return
        now = rospy.Time.now()
        with self.lock:
            odometry = self.odometry
            flight_state = self.flight_state
            command = self.command
        if odometry is None or now.is_zero():
            return
        if flight_state != FLYING:
            self.flying_since = None
            self.takeoff_pub.publish(Empty())
            return
        if self.flying_since is None:
            self.flying_since = now
            return
        if self.start_time is None:
            if (now-self.flying_since).to_sec() < self.settle_time:
                return
            self.start_time = now
            self.enable_pub.publish(Bool(data=True))
            rospy.loginfo("Velocity-response sequence started: %s", self.run_id)

        elapsed = (now-self.start_time).to_sec()
        phase, phase_time, policy = self._phase(elapsed)
        self._publish_policy(now, policy)
        if phase is None:
            self._finish()
            return

        pose = odometry.pose.pose
        actual = rotate_vector(pose.orientation, odometry.twist.twist.linear)
        row = {
            "time": elapsed,
            "phase": phase,
            "phase_time": phase_time,
            "x": pose.position.x,
            "y": pose.position.y,
            "z": pose.position.z,
            "actual_vx": actual[0],
            "actual_vy": actual[1],
            "actual_vz": actual[2],
            "actual_horizontal_speed": math.hypot(actual[0], actual[1]),
            "policy_vx": policy[0],
            "policy_vy": policy[1],
            "policy_vz": policy[2],
            "command_vx": None if command is None else command.twist.linear.x,
            "command_vy": None if command is None else command.twist.linear.y,
            "command_vz": None if command is None else command.twist.linear.z,
        }
        self.samples.append(row)

    def _finish(self):
        if self.done:
            return
        self.done = True
        self._stop()
        trajectory = write_trajectory(
            str(self.output_directory), "velocity_response", self.samples, self.run_id
        )
        result = compute_response_metrics(self.samples)
        result.update({
            "run_id": self.run_id,
            "controller_config": self.controller_config,
            "trajectory_file": str(trajectory),
        })
        self.output_directory.mkdir(parents=True, exist_ok=True)
        destination = self.output_directory/f"velocity_response__{self.run_id}.json"
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{destination.stem}.", suffix=".tmp", dir=str(self.output_directory)
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(result, stream, indent=2, sort_keys=True)
                stream.write("\n")
            os.replace(temporary, destination)
        except Exception:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise
        rospy.loginfo("Velocity-response result: %s", destination)
        rospy.signal_shutdown("velocity-response sequence complete")

    def _stop(self):
        self.enable_pub.publish(Bool(data=False))
        if self.start_time is not None:
            self._publish_policy(rospy.Time.now(), (0.0, 0.0, 0.0))


def main():
    rospy.init_node("dnav_velocity_response")
    VelocityResponseRunner()
    rospy.spin()


if __name__ == "__main__":
    main()
