cmake_minimum_required(VERSION 3.16)
project(go2_parkour_deploy LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(Eigen3 REQUIRED)
find_package(fmt REQUIRED)
find_package(onnxruntime REQUIRED)
find_package(unitree_sdk2 REQUIRED)
find_package(ZLIB REQUIRED)
find_package(Threads REQUIRED)
find_library(ZMQ_LIBRARY NAMES zmq REQUIRED)

add_executable(deploy src/deploy.cpp)
target_include_directories(deploy PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(deploy PRIVATE
  Eigen3::Eigen
  fmt::fmt
  onnxruntime
  unitree_sdk2
  ZLIB::ZLIB
  Threads::Threads
  ${ZMQ_LIBRARY}
)

include(CTest)
if(BUILD_TESTING)
  add_executable(depth_frame_history_test tests/depth_frame_history_test.cpp)
  target_include_directories(depth_frame_history_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  add_test(NAME depth_frame_history COMMAND depth_frame_history_test)
endif()

install(TARGETS deploy
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY model
  DESTINATION lib/${PROJECT_NAME}
  OPTIONAL
)
