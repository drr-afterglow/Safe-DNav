cmake_minimum_required(VERSION 3.0.2)
project(dnav_evaluation)

find_package(catkin REQUIRED COMPONENTS
  dnav_perception
  diagnostic_msgs
  gazebo_msgs
  geometry_msgs
  nav_msgs
  rosbag
  rospy
  sensor_msgs
  std_msgs
)

catkin_python_setup()

catkin_package(CATKIN_DEPENDS
  dnav_perception
  diagnostic_msgs
  gazebo_msgs
  geometry_msgs
  nav_msgs
  rosbag
  rospy
  sensor_msgs
  std_msgs
)

install(DIRECTORY config launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

catkin_install_python(PROGRAMS
  scripts/dnav_episode_runner.py
  scripts/dnav_velocity_response.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

if(CATKIN_ENABLE_TESTING)
  catkin_add_nosetests(test)
endif()
