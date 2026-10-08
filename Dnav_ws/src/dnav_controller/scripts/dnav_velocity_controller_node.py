#!/usr/bin/env python3
import math
import rospy
from geometry_msgs.msg import TwistStamped, Vector3Stamped
from nav_msgs.msg import Odometry

class Adapter:
    def __init__(self):
        self.yaw = 0.0; self.cmd = None; self.last = rospy.Time(0)
        self.pub = rospy.Publisher('/dnav/uav/cmd_vel', TwistStamped, queue_size=10)
        rospy.Subscriber('~world_velocity', Vector3Stamped, self.velocity)
        rospy.Subscriber('/dnav/uav/odom_raw', Odometry, self.odom)
        self.timer = rospy.Timer(rospy.Duration(0.02), self.tick)
    def velocity(self, m): self.cmd=m.vector; self.last=rospy.Time.now()
    def odom(self,m):
        q=m.pose.pose.orientation
        self.yaw=math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))
    def tick(self,_):
        out=TwistStamped(); out.header.stamp=rospy.Time.now(); out.header.frame_id='base_link'
        if self.cmd and (rospy.Time.now()-self.last).to_sec() < 0.5:
            c,s=math.cos(self.yaw),math.sin(self.yaw)
            out.twist.linear.x=max(-3,min(3,c*self.cmd.x+s*self.cmd.y))
            out.twist.linear.y=max(-3,min(3,-s*self.cmd.x+c*self.cmd.y))
            out.twist.linear.z=max(-3,min(3,self.cmd.z))
        self.pub.publish(out)
if __name__=='__main__':
    rospy.init_node('dnav_velocity_controller'); Adapter(); rospy.spin()
