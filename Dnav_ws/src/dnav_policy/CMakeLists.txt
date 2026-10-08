cmake_minimum_required(VERSION 3.0.2)
project(dnav_policy)
find_package(catkin REQUIRED COMPONENTS
  diagnostic_msgs
  dnav_perception
  geometry_msgs
  nav_msgs
  rospy
  std_msgs
)
catkin_python_setup()
catkin_package(CATKIN_DEPENDS diagnostic_msgs dnav_perception geometry_msgs nav_msgs rospy std_msgs)
catkin_install_python(PROGRAMS scripts/dnav_policy_node.py DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
install(DIRECTORY config launch DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
