cmake_minimum_required(VERSION 3.0.2)
project(dnav_controller)
find_package(catkin REQUIRED COMPONENTS diagnostic_msgs dnav_perception geometry_msgs nav_msgs roscpp sensor_msgs std_msgs std_srvs)
catkin_package(INCLUDE_DIRS include LIBRARIES dnav_safety_controller CATKIN_DEPENDS diagnostic_msgs dnav_perception geometry_msgs nav_msgs roscpp sensor_msgs std_msgs std_srvs)
include_directories(${catkin_INCLUDE_DIRS})
include_directories(include)
add_library(dnav_safety_controller
  src/safety_controller.cpp
  src/dynamic_obstacle_cluster.cpp
  src/obstacle_safety_filter.cpp)
target_link_libraries(dnav_safety_controller ${catkin_LIBRARIES})
add_executable(dnav_velocity_controller_node src/dnav_velocity_controller_node.cpp)
target_link_libraries(dnav_velocity_controller_node dnav_safety_controller ${catkin_LIBRARIES})
add_executable(dnav_obstacle_safety_filter_node src/dnav_obstacle_safety_filter_node.cpp)
target_link_libraries(dnav_obstacle_safety_filter_node dnav_safety_controller ${catkin_LIBRARIES})
install(TARGETS dnav_safety_controller dnav_velocity_controller_node dnav_obstacle_safety_filter_node
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
install(DIRECTORY include/${PROJECT_NAME}/ DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION})
install(DIRECTORY config launch DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})

if(CATKIN_ENABLE_TESTING)
  catkin_add_gtest(test_safety_controller test/test_safety_controller.cpp)
  if(TARGET test_safety_controller)
    target_link_libraries(test_safety_controller dnav_safety_controller ${catkin_LIBRARIES})
  endif()
  catkin_add_gtest(test_obstacle_safety_filter test/test_obstacle_safety_filter.cpp)
  if(TARGET test_obstacle_safety_filter)
    target_link_libraries(test_obstacle_safety_filter dnav_safety_controller ${catkin_LIBRARIES})
  endif()
  catkin_add_gtest(test_dynamic_obstacle_cluster test/test_dynamic_obstacle_cluster.cpp)
  if(TARGET test_dynamic_obstacle_cluster)
    target_link_libraries(test_dynamic_obstacle_cluster dnav_safety_controller ${catkin_LIBRARIES})
  endif()
endif()
