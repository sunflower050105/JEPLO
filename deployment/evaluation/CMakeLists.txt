cmake_minimum_required(VERSION 3.16)
project(jeplo_sim_evaluation LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_library(ZMQ_LIBRARY zmq REQUIRED)

# Uses the same projection and accumulation code as the ROS publisher, with
# its simulation path compiled without ROS/Livox dependencies.
add_executable(lidar_depth_sim ../lidar_depth_pub/lidar_depth_pub.cpp)
target_compile_definitions(lidar_depth_sim PRIVATE JEPLO_SIM_ONLY)
target_link_libraries(lidar_depth_sim PRIVATE ${ZMQ_LIBRARY} pthread)
target_compile_options(lidar_depth_sim PRIVATE -Wall -Wextra)

include(CTest)
if(BUILD_TESTING)
  find_path(MUJOCO_INCLUDE_DIR mujoco/mujoco.h
            PATHS ../unitree_mujoco/simulate/mujoco/include NO_DEFAULT_PATH REQUIRED)
  find_library(MUJOCO_LIBRARY mujoco
               PATHS ../unitree_mujoco/simulate/mujoco/lib NO_DEFAULT_PATH REQUIRED)
  add_executable(test_see_then_hidden tests/test_see_then_hidden.cpp)
  target_include_directories(test_see_then_hidden PRIVATE ${MUJOCO_INCLUDE_DIR})
  target_link_libraries(test_see_then_hidden PRIVATE ${MUJOCO_LIBRARY})
  target_compile_options(test_see_then_hidden PRIVATE -Wall -Wextra)
  add_test(NAME see_then_hidden COMMAND test_see_then_hidden)
endif()
