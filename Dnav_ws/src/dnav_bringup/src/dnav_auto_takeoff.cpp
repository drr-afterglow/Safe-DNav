#include <ros/ros.h>
#include <std_msgs/Empty.h>
int main(int argc, char** argv) { ros::init(argc, argv, "dnav_auto_takeoff"); ros::NodeHandle nh; ros::Publisher p=nh.advertise<std_msgs::Empty>("/dnav/uav/takeoff",1); ros::Duration(5.0).sleep(); ros::Rate rate(10); for (int i=0; i<20 && ros::ok(); ++i) { p.publish(std_msgs::Empty()); ros::spinOnce(); rate.sleep(); } return 0; }
