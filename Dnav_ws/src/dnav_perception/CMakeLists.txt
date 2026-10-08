cmake_minimum_required(VERSION 3.0.2)
project(dnav_perception)

find_package(catkin REQUIRED COMPONENTS
  diagnostic_msgs
  message_generation
  nav_msgs
  rospy
  sensor_msgs
  std_msgs
  tf2_ros
)

catkin_python_setup()
add_message_files(FILES PerceptionObservation.msg)
generate_messages(DEPENDENCIES std_msgs)
catkin_package(CATKIN_DEPENDS diagnostic_msgs message_runtime nav_msgs rospy sensor_msgs std_msgs tf2_ros)

catkin_install_python(PROGRAMS scripts/dnav_perception_node.py DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

install(DIRECTORY config launch models
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
