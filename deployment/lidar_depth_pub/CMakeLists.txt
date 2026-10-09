cmake_minimum_required(VERSION 3.16)
project(lidar_depth_pub)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Dependencies ──────────────────────────────────────────────────────────────
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(livox_ros_driver2 REQUIRED)

# libzmq — try pkg-config first, fall back to direct linking
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(ZMQ IMPORTED_TARGET libzmq)
endif()

# ── Target ────────────────────────────────────────────────────────────────────
set(LIDAR_DEPTH_TARGETS lidar_depth_pub)
add_executable(lidar_depth_pub lidar_depth_pub.cpp)

# lidar_depth_pub_occ.cpp is not shipped in the public release; build it only if present
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/lidar_depth_pub_occ.cpp)
    list(APPEND LIDAR_DEPTH_TARGETS lidar_depth_pub_occ)
    add_executable(lidar_depth_pub_occ lidar_depth_pub_occ.cpp)
endif()

foreach(target ${LIDAR_DEPTH_TARGETS})
    ament_target_dependencies(${target}
        rclcpp
        livox_ros_driver2
    )

    target_link_libraries(${target}
        $<IF:$<TARGET_EXISTS:PkgConfig::ZMQ>,PkgConfig::ZMQ,zmq>
        pthread
    )

    target_compile_options(${target} PRIVATE -Wall -Wextra -O2)
endforeach()

# ── Install ───────────────────────────────────────────────────────────────────
install(TARGETS ${LIDAR_DEPTH_TARGETS} DESTINATION lib/${PROJECT_NAME})

ament_package()
