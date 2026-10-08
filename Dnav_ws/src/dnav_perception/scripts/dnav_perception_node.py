#!/usr/bin/env python3
"""ROS1 deployment node for learned lidar perception."""

from pathlib import Path
from threading import Lock, Thread
import time

import numpy as np
import rospkg
import rospy
import tf2_ros
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, LaserScan

from dnav_perception.geometry import PanoramaSpec
from dnav_perception.msg import PerceptionObservation
from dnav_perception.pose_buffer import PoseBuffer, PoseSample
from dnav_perception.runtime import PerceptionRuntime
from dnav_perception.scan_adapter import ScanSpec, ranges_to_sensor_points, transform_points


class PerceptionNode:
    def __init__(self):
        package_path = Path(rospkg.RosPack().get_path("dnav_perception"))
        model_setting = rospy.get_param("~model_file", "models/perception_best.ts.pt")
        model_path = Path(model_setting)
        if not model_path.is_absolute():
            model_path = package_path / model_path

        height = int(rospy.get_param("~height", 16))
        width = int(rospy.get_param("~width", 180))
        max_range = float(rospy.get_param("~max_range", 10.0))
        vfov = (
            float(rospy.get_param("~vertical_min_deg", -7.0)),
            float(rospy.get_param("~vertical_max_deg", 52.0)),
        )
        self.scan_spec = ScanSpec(
            int(rospy.get_param("~scan_height", 16)),
            int(rospy.get_param("~scan_width", 360)),
            max_range,
            float(rospy.get_param("~sensor_vertical_min_deg", -7.0)),
            float(rospy.get_param("~sensor_vertical_max_deg", 52.0)),
        )
        self.runtime = PerceptionRuntime(
            str(model_path),
            PanoramaSpec(height, width, max_range, vfov),
            device=rospy.get_param("~device", "cuda"),
            reference_delta_t=float(rospy.get_param("~reference_delta_t", 0.096)),
            min_delta_t=float(rospy.get_param("~min_delta_t", 0.05)),
            max_delta_t=float(rospy.get_param("~max_delta_t", 0.15)),
            mask_threshold=float(rospy.get_param("~mask_threshold", 0.50)),
            dyn_max_vel=float(rospy.get_param("~dyn_max_vel", 5.0)),
            output_size=(int(rospy.get_param("~rl_height", 6)), int(rospy.get_param("~rl_width", 36))),
        )
        self.body_frame = rospy.get_param("~body_frame", "base_link")
        self.sensor_frame = rospy.get_param("~sensor_frame", "mid360_link")
        self.pose_tolerance = float(rospy.get_param("~pose_tolerance", 0.05))
        self.tf_timeout = float(rospy.get_param("~tf_timeout", 0.10))
        self.worker_rate = float(rospy.get_param("~worker_rate", 200.0))
        self.pose_buffer = PoseBuffer()
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(5.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.pending_lock = Lock()
        self.pending_scan = None
        self.running = True
        self.last_scan_stamp = None
        self.drop_count = 0
        self.error_count = 0
        self.last_latency_ms = 0.0
        self.publish_debug_images = bool(rospy.get_param("~publish_debug_images", True))

        self.publisher = rospy.Publisher(
            rospy.get_param("~output_topic", "/dnav/perception/observation"),
            PerceptionObservation,
            queue_size=1,
        )
        self.diagnostic_publisher = rospy.Publisher(
            rospy.get_param("~diagnostics_topic", "/dnav/perception/diagnostics"),
            DiagnosticArray,
            queue_size=1,
        )
        self.depth_debug_publisher = rospy.Publisher(
            rospy.get_param("~depth_debug_topic", "/dnav/perception/debug/depth"), Image, queue_size=1
        )
        self.probability_debug_publisher = rospy.Publisher(
            rospy.get_param("~dynamic_probability_topic", "/dnav/perception/debug/dynamic_probability"),
            Image,
            queue_size=1,
        )
        self.odom_subscriber = rospy.Subscriber(
            rospy.get_param("~odom_topic", "/dnav/uav/odom_raw"),
            Odometry,
            self._odom_callback,
            queue_size=100,
        )
        self.scan_subscriber = rospy.Subscriber(
            rospy.get_param("~scan_topic", "/dnav/uav/mid360/scan"),
            LaserScan,
            self._scan_callback,
            queue_size=1,
        )
        self.worker = Thread(target=self._worker_loop, name="dnav-perception-worker", daemon=True)
        self.worker.start()
        rospy.on_shutdown(self.shutdown)
        rospy.loginfo("Perception model loaded on %s: %s", self.runtime.device, model_path)

    def _odom_callback(self, message):
        position = message.pose.pose.position
        quaternion = message.pose.pose.orientation
        velocity = message.twist.twist.linear
        sample = PoseSample(
            message.header.stamp.to_sec(),
            np.array([position.x, position.y, position.z], dtype=np.float64),
            np.array([quaternion.w, quaternion.x, quaternion.y, quaternion.z], dtype=np.float64),
            np.array([velocity.x, velocity.y, velocity.z], dtype=np.float64),
        )
        try:
            self.pose_buffer.add(sample)
        except ValueError as error:
            rospy.logwarn_throttle(2.0, "Rejected odometry: %s", error)

    def _scan_callback(self, message):
        with self.pending_lock:
            if self.pending_scan is not None:
                self.drop_count += 1
            self.pending_scan = (message, time.monotonic())

    def _worker_loop(self):
        rate = rospy.Rate(self.worker_rate)
        while self.running and not rospy.is_shutdown():
            try:
                self._process_pending()
            except Exception as error:
                self.error_count += 1
                self.runtime.reset()
                rospy.logerr_throttle(1.0, "Perception frame failed: %s", error)
                self._publish_diagnostic(DiagnosticStatus.ERROR, str(error))
            try:
                rate.sleep()
            except rospy.exceptions.ROSInterruptException:
                if rospy.is_shutdown() or not self.running:
                    break
                raise
            except rospy.exceptions.ROSTimeMovedBackwardsException as error:
                # Gazebo reset rewinds /clock. rospy.Rate keeps its previous
                # deadline and raises, so rebuild it and discard all temporal
                # data instead of letting the worker thread terminate.
                self.runtime.reset()
                self.pose_buffer.clear()
                with self.pending_lock:
                    self.pending_scan = None
                self.last_scan_stamp = None
                rate = rospy.Rate(self.worker_rate)
                rospy.logwarn("ROS time moved backwards; perception history reset: %s", error)

    def _process_pending(self):
        with self.pending_lock:
            pending = self.pending_scan
        if pending is None:
            return
        message, arrival = pending
        stamp = message.header.stamp.to_sec()
        pose = self.pose_buffer.interpolate(stamp, self.pose_tolerance)
        if pose is None:
            if time.monotonic() - arrival < max(0.2, 2.0 * self.pose_tolerance):
                return
            with self.pending_lock:
                if self.pending_scan is pending:
                    self.pending_scan = None
            self.drop_count += 1
            self.runtime.reset()
            self._publish_diagnostic(DiagnosticStatus.WARN, "odometry unavailable at scan timestamp")
            return
        source_frame = message.header.frame_id or self.sensor_frame
        if not self.tf_buffer.can_transform(self.body_frame, source_frame, rospy.Time(0)):
            if time.monotonic() - arrival < max(0.2, 2.0 * self.tf_timeout):
                return
            with self.pending_lock:
                if self.pending_scan is pending:
                    self.pending_scan = None
            self.drop_count += 1
            self.runtime.reset()
            self._publish_diagnostic(DiagnosticStatus.WARN, "sensor extrinsic transform unavailable")
            return
        with self.pending_lock:
            if self.pending_scan is not pending:
                return
            self.pending_scan = None

        if self.last_scan_stamp is not None and stamp <= self.last_scan_stamp:
            self.runtime.reset()
            self.pose_buffer.clear()
        self.last_scan_stamp = stamp
        started = time.perf_counter()
        points, valid = ranges_to_sensor_points(
            message.ranges,
            message.angle_min,
            message.angle_increment,
            message.range_min,
            message.range_max,
            self.scan_spec,
        )
        transform = self.tf_buffer.lookup_transform(
            self.body_frame,
            source_frame,
            # This is a rigid calibration transform. setupTF.launch currently
            # publishes it on /tf rather than /tf_static, so querying the scan
            # timestamp would spuriously require temporal extrapolation.
            rospy.Time(0),
            rospy.Duration(self.tf_timeout),
        ).transform
        translation = np.array([transform.translation.x, transform.translation.y, transform.translation.z])
        rotation = np.array(
            [transform.rotation.w, transform.rotation.x, transform.rotation.y, transform.rotation.z]
        )
        points = transform_points(points, translation, rotation)
        static, dynamic, history_valid, delta_t = self.runtime.process(
            points,
            valid,
            stamp,
            pose.position,
            pose.quaternion_wxyz,
            pose.linear_velocity_body,
        )
        output = PerceptionObservation()
        output.header = message.header
        output.header.frame_id = self.body_frame
        output.history_valid = history_valid
        output.delta_t = delta_t
        output.lidar_static = static.astype(np.float32).reshape(-1).tolist()
        output.lidar_dynamic = dynamic.astype(np.float32).reshape(-1).tolist()
        self.publisher.publish(output)
        if self.publish_debug_images:
            self._publish_debug_image(
                self.depth_debug_publisher, message.header, self.runtime.last_depth_image
            )
            self._publish_debug_image(
                self.probability_debug_publisher,
                message.header,
                self.runtime.last_probability_image,
            )
        self.last_latency_ms = (time.perf_counter() - started) * 1000.0
        self._publish_diagnostic(DiagnosticStatus.OK, "perception active")

    @staticmethod
    def _publish_debug_image(publisher, header, normalized_image):
        image = Image()
        image.header = header
        image.height, image.width = normalized_image.shape
        image.encoding = "mono8"
        image.is_bigendian = False
        image.step = image.width
        image.data = np.rint(np.clip(normalized_image, 0.0, 1.0) * 255.0).astype(np.uint8).tobytes()
        publisher.publish(image)

    def _publish_diagnostic(self, level, text):
        array = DiagnosticArray()
        array.header.stamp = rospy.Time.now()
        status = DiagnosticStatus(level=level, name="dnav_perception", message=text, hardware_id="mid360")
        status.values = [
            KeyValue("device", str(self.runtime.device)),
            KeyValue("history_valid", str(self.runtime.previous_points is not None).lower()),
            KeyValue("deskew", "false"),
            KeyValue("latency_ms", f"{self.last_latency_ms:.3f}"),
            KeyValue("dropped_scans", str(self.drop_count)),
            KeyValue("errors", str(self.error_count)),
        ]
        array.status = [status]
        self.diagnostic_publisher.publish(array)

    def shutdown(self):
        self.running = False
        if self.worker.is_alive():
            self.worker.join(timeout=1.0)


def main():
    rospy.init_node("dnav_perception")
    PerceptionNode()
    rospy.spin()


if __name__ == "__main__":
    main()
