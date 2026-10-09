#pragma once

#include <boost/program_options.hpp>
#include <filesystem>
#include <iostream>
#include <yaml-cpp/yaml.h>

namespace param {

inline struct SimulationConfig {
    std::string robot;
    std::filesystem::path robot_scene;

    int domain_id;
    std::string interface;

    int use_joystick;
    std::string joystick_type;
    std::string joystick_device;
    int joystick_bits;

    int print_scene_information;

    int enable_elastic_band;
    int band_attached_link = 0;

    int enable_depth = 0;

    int enable_lidar = 0;
    bool lidar_legacy_fov = false;
    std::filesystem::path lidar_pattern;
    double lidar_hz = 20.0;
    int lidar_port = 5590;
    bool see_then_hidden = false;
    double see_then_hidden_distance = 1.2;
    int see_then_hidden_preview_scans = 3;
    std::string see_then_hidden_target = "box1";
    std::string see_then_hidden_log;

    void load_from_yaml(const std::string &filename) {
        auto cfg = YAML::LoadFile(filename);
        try {
            robot = cfg["robot"].as<std::string>();
            robot_scene = cfg["robot_scene"].as<std::string>();
            domain_id = cfg["domain_id"].as<int>();
            interface = cfg["interface"].as<std::string>();
            use_joystick = cfg["use_joystick"].as<int>();
            joystick_type = cfg["joystick_type"].as<std::string>();
            joystick_device = cfg["joystick_device"].as<std::string>();
            joystick_bits = cfg["joystick_bits"].as<int>();
            print_scene_information = cfg["print_scene_information"].as<int>();
            enable_elastic_band = cfg["enable_elastic_band"].as<int>();
        } catch (const std::exception &e) {
            std::cerr << e.what() << '\n';
            exit(EXIT_FAILURE);
        }
    }
} config;

/* ---------- Command Line Parameters ---------- */
namespace po = boost::program_options;

// ※ This function must be called at the beginning of main() function
inline po::variables_map helper(int argc, char **argv) {
    po::options_description desc("Unitree Mujoco");
    desc.add_options()("help,h", "Show help message")(
        "domain_id,i", po::value<int>(&config.domain_id), "DDS domain ID; -i 0")(
        "network,n", po::value<std::string>(&config.interface), "DDS network interface; -n eth0")(
        "robot,r", po::value<std::string>(&config.robot), "Robot type; -r go2")(
        "scene,s", po::value<std::filesystem::path>(&config.robot_scene),
        "Robot scene file; -s scene_terrain.xml")("depth,d", "Enable depth image publishing")(
        "lidar,l", "Enable Mid360 point cloud publishing for lidar_depth_pub --sim")(
        "lidar-legacy-fov", "Use the lidar_depth_pub 25x60 legacy FOV")(
        "lidar-pattern", po::value<std::filesystem::path>(&config.lidar_pattern),
        "Path to the Mid360 scan_mode/mid360.npy file")(
        "lidar-hz", po::value<double>(&config.lidar_hz), "LiDAR scan rate (default: 20 Hz)")(
        "lidar-port", po::value<int>(&config.lidar_port), "LiDAR ZMQ port (default: 5590)")(
        "see-then-hidden", "Sim-only target-region occlusion after a clean preview")(
        "see-then-hidden-distance", po::value<double>(&config.see_then_hidden_distance),
        "Target-surface distance in metres for occlusion onset (default: 1.2)")(
        "see-then-hidden-preview-scans", po::value<int>(&config.see_then_hidden_preview_scans),
        "Consecutive LiDAR scans that must see the target first (default: 3)")(
        "see-then-hidden-target", po::value<std::string>(&config.see_then_hidden_target),
        "MuJoCo box geom or body of box geoms to hide (e.g. box1 or stairs)")(
        "see-then-hidden-log", po::value<std::string>(&config.see_then_hidden_log),
        "Write per-scan diagnostic CSV to this path");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << std::endl;
        exit(0);
    }

    if (vm.count("depth")) {
        config.enable_depth = 1;
    }
    if (vm.count("lidar")) {
        config.enable_lidar = 1;
    }
    if (vm.count("lidar-legacy-fov")) {
        config.lidar_legacy_fov = true;
    }
    if (vm.count("see-then-hidden")) {
        config.see_then_hidden = true;
        if (!config.enable_lidar || config.see_then_hidden_distance <= 0.0 ||
            config.see_then_hidden_preview_scans < 1) {
            std::cerr << "--see-then-hidden requires --lidar, a positive distance, "
                         "and at least one preview scan\n";
            exit(EXIT_FAILURE);
        }
    }

    return vm;
}

} // namespace param
