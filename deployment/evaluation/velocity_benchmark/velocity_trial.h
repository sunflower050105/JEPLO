#pragma once

#include <mujoco/mujoco.h>
#include "box_ascent.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

// Ground truth for scoring and later video rendering; never a policy input.
namespace velocity_benchmark {
struct Config {
    std::string directory;
    std::string terrain = "flat";
    double duration = 8.0;
    double speed = 0.8;
    double min_progress_ratio = 0.8;
    std::string box_task = "ascent";
    double stair_contact_friction = 0.4;
};

class Trial {
  public:
    explicit Trial(Config config) : config_(std::move(config)) {}

    // Called with the simulation mutex held, after stepping and updating terrain.
    bool update(mjModel *m, const mjData *d) {
        if (finished_) return true;
        if (!initialized_) initialize(m, d);
        if (d->time < last_sim_time_)
            throw std::runtime_error("Velocity benchmark: reset during trial is not allowed");
        last_sim_time_ = d->time;
        const double *p = d->xpos + 3 * base_id_;
        const double *q = d->xquat + 4 * base_id_;
        const double *r = d->xmat + 9 * base_id_;
        const double tilt = std::acos(std::clamp(r[8], -1.0, 1.0)) * kDeg;
        const double yaw = std::atan2(2 * (q[0] * q[3] + q[1] * q[2]),
                                    1 - 2 * (q[2] * q[2] + q[3] * q[3])) * kDeg;
        const double roll = std::atan2(2 * (q[0] * q[1] + q[2] * q[3]),
                                     1 - 2 * (q[1] * q[1] + q[2] * q[2])) * kDeg;
        const double pitch = std::asin(std::clamp(2 * (q[0] * q[2] - q[3] * q[1]),
                                                -1.0, 1.0)) * kDeg;
        const auto wall = std::chrono::system_clock::now().time_since_epoch();
        const auto wall_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(wall).count();

        // Atomic heartbeat lets the runner wait in simulation time and detect a paused UI.
        if (d->time >= next_status_) {
            std::ofstream status(path("status.tmp"));
            status << std::setprecision(12) << "{\"sim_time_s\":" << d->time
                   << ",\"x_m\":" << p[0] << ",\"y_m\":" << p[1]
                   << ",\"z_m\":" << p[2] << ",\"yaw_deg\":" << yaw
                   << ",\"tilt_deg\":" << tilt;
            if (config_.terrain == "benchmark_stairs")
                status << ",\"course_center_y_m\":" << course_center_y_;
            status << ",\"armed\":"
                   << (armed_ ? "true" : "false") << "}\n";
            status.close();
            std::filesystem::rename(path("status.tmp"), path("status.json"));
            next_status_ = d->time + 0.05;
        }
        if (!armed_) {
            if (!std::filesystem::exists(path("GO"))) return false;
            t0_ = d->time;
            wall0_ns_ = wall_ns;
            x0_ = p[0];
            y0_ = p[1];
            // Axis stays world +X, the same fixed course direction in both runs.
            goal_x_ = x0_ + config_.min_progress_ratio * config_.speed * config_.duration;
            if (config_.terrain == "boxes") {
                const int box = mj_name2id(m, mjOBJ_GEOM, "box2");
                if (box < 0) throw std::runtime_error("Benchmark boxes require box2");
                if (config_.box_task == "traverse")
                    goal_x_ = d->geom_xpos[3 * box] + m->geom_size[3 * box] + 0.4;
                else
                    ascent_ = std::make_unique<BoxAscent>(m);
            } else if (config_.terrain == "benchmark_stairs") {
                ascent_ = std::make_unique<BoxAscent>(m, "benchmark_stair_landing");
            }
            mj_saveModel(m, path("scene.mjb").c_str(), nullptr, 0);
            csv_.open(path("state.csv"));
            if (!csv_) throw std::runtime_error("Cannot create benchmark state.csv");
            csv_ << "wall_ns,sim_time_s,trial_time_s,x_m,y_m,z_m,forward_m,"
                    "forward_speed_mps,roll_deg,pitch_deg,yaw_deg,tilt_deg,fall_condition";
            for (int i = 0; i < m->nq; ++i) csv_ << ",qpos_" << i;
            csv_ << ",ascent_on_top_mask,ascent_contact_mask,ascent_recent_contact_mask,"
                    "ascent_candidate,ascent_hold_s";
            csv_ << '\n';
            armed_ = true;
            last_x_ = x0_;
            std::cout << "[VelocityTrial] ARMED speed=" << config_.speed
                      << " scoring_rule=" << scoringRule() << std::endl;
        }

        const double elapsed = d->time - t0_;
        if (elapsed + 1e-9 < next_sample_) return false;
        const double dt = elapsed - last_sample_;
        const double velocity = dt > 0 ? (p[0] - last_x_) / dt : 0;
        const bool fall = p[2] < 0.16 || tilt > 60;
        if (dt > 0) {
            error_integral_ += std::pow(velocity - config_.speed, 2) * dt;
            fall_duration_ = fall ? fall_duration_ + dt : 0;
        }
        max_tilt_ = std::max(max_tilt_, tilt);
        const BoxAscent::State ascent = ascent_ ? ascent_->update(m, d, elapsed) : BoxAscent::State{};
        csv_ << std::setprecision(12) << wall_ns << ',' << d->time << ',' << elapsed
             << ',' << p[0] << ',' << p[1] << ',' << p[2] << ',' << p[0] - x0_
             << ',' << velocity << ',' << roll << ',' << pitch << ',' << yaw
             << ',' << tilt << ',' << fall;
        for (int i = 0; i < m->nq; ++i) csv_ << ',' << d->qpos[i];
        csv_ << ',' << ascent.on_top_mask << ',' << ascent.contact_mask
             << ',' << ascent.recent_contact_mask << ',' << ascent.candidate << ',' << ascent.hold_s;
        csv_ << '\n';
        // Flush a small 50 Hz trace so it survives an interrupted process.
        csv_.flush();
        last_sample_ = elapsed;
        last_x_ = p[0];
        next_sample_ = elapsed + 0.02;

        // A corridor prevents going around an obstacle from counting as completion.
        const double corridor = config_.terrain == "boxes" ? 0.5 :
                                config_.terrain == "benchmark_stairs" ? 0.8 : 1.0;
        if (p[0] >= goal_x_ && goal_time_ < 0) goal_time_ = elapsed;
        const bool boxes_complete = config_.terrain == "boxes" &&
            (config_.box_task == "ascent" ?
                ascent.achieved_at >= 0 && ascent.achieved_at <= config_.duration + 1e-9 :
                goal_time_ >= 0 && elapsed - goal_time_ >= 1.0 && p[0] >= goal_x_);
        const bool stairs_complete = config_.terrain == "benchmark_stairs" &&
            ascent.achieved_at >= 0 && ascent.achieved_at <= config_.duration + 1e-9;
        std::string outcome;
        if (fall_duration_ >= 0.2) outcome = "fall";
        else if (std::abs(p[1]) > corridor) outcome = "left_course";
        else if (boxes_complete || stairs_complete) outcome = "success";
        else if (elapsed >= config_.duration) {
            outcome = config_.terrain == "flat" && p[0] >= goal_x_ ? "success" : "timeout";
        }
        if (!outcome.empty()) {
            const double wall_duration = (wall_ns - wall0_ns_) / 1e9;
            std::ofstream result(path("result.tmp"));
            result << std::setprecision(12)
                   << "{\n  \"terrain\": \"" << config_.terrain << "\",\n"
                   << "  \"box_task\": " << (config_.terrain == "boxes" ?
                       "\"" + config_.box_task + "\"" : "null") << ",\n"
                   << "  \"scoring_rule\": \"" << scoringRule() << "\",\n"
                   << "  \"mujoco_version\": \"" << mj_versionString() << "\",\n"
                   << "  \"command_mps\": " << config_.speed << ",\n"
                   << "  \"stair_contact_friction\": "
                   << (config_.terrain == "benchmark_stairs" ? config_.stair_contact_friction : 0.0)
                   << ",\n"
                   << "  \"duration_s\": " << elapsed << ",\n"
                   << "  \"wall_duration_s\": " << wall_duration << ",\n"
                   << "  \"real_time_factor\": " << elapsed / std::max(wall_duration, 1e-9) << ",\n"
                   << "  \"start_x_m\": " << x0_ << ",\n"
                   << "  \"start_y_m\": " << y0_ << ",\n"
                   << "  \"progress_m\": " << p[0] - x0_ << ",\n"
                   << "  \"goal_x_m\": ";
            if (ascent_) result << "null";  // Ascent has no scalar X finish line.
            else result << goal_x_;
            result << ",\n  \"ascent_time_s\": ";
            if (ascent_ && outcome == "success") result << ascent.achieved_at;
            else result << "null";
            if (ascent_) {
                result << ",\n  \"ascent_target_geom\": \""
                       << (config_.terrain == "benchmark_stairs" ? "benchmark_stair_landing" : "box2")
                       << "\",\n  \"ascent_criteria\": {"
                       << "\"contact_window_s\":" << BoxAscent::kContactWindow
                       << ",\"hold_s\":" << BoxAscent::kHold
                       << ",\"max_sample_gap_s\":" << BoxAscent::kMaxSampleGap
                       << ",\"max_tilt_deg\":" << BoxAscent::kMaxTiltDeg
                       << ",\"min_base_clearance_m\":" << BoxAscent::kMinClearance
                       << ",\"top_tolerance_m\":" << BoxAscent::kTopTolerance
                       << ",\"contact_distance_m\":" << BoxAscent::kContactDistance
                       << ",\"min_normal_alignment\":" << BoxAscent::kMinNormalAlignment << '}';
            }
            result << ",\n"
                   << "  \"mean_speed_mps\": " << (elapsed > 0 ? (p[0] - x0_) / elapsed : 0) << ",\n"
                   << "  \"speed_rmse_mps\": " << (elapsed > 0 ? std::sqrt(error_integral_ / elapsed) : 0) << ",\n"
                   << "  \"max_tilt_deg\": " << max_tilt_ << ",\n"
                   << "  \"success\": " << (outcome == "success" ? "true" : "false")
                   << ",\n  \"outcome\": \"" << outcome << "\"\n}\n";
            result.close();
            std::filesystem::rename(path("result.tmp"), path("result.json"));
            std::cout << "[VelocityTrial] " << outcome << std::endl;
            finished_ = true;
        }
        return finished_;
    }

  private:
    const char *scoringRule() const {
        if (config_.terrain == "flat") return "flat_progress_v1";
        if (config_.terrain == "benchmark_stairs") return "stair_landing_v1";
        return config_.box_task == "ascent" ? "box_ascent_v2" : "box_traverse_v1";
    }
    std::string path(const char *name) const {
        return (std::filesystem::path(config_.directory) / name).string();
    }
    void initialize(mjModel *m, const mjData *d) {
        if (config_.terrain != "flat" && config_.terrain != "boxes" &&
            config_.terrain != "benchmark_stairs")
            throw std::runtime_error("Unknown benchmark terrain");
        if (config_.box_task != "ascent" && config_.box_task != "traverse")
            throw std::runtime_error("Unknown benchmark box task");
        base_id_ = mj_name2id(m, mjOBJ_BODY, "base_link");
        if (base_id_ < 0) throw std::runtime_error("Velocity benchmark requires Go2 base_link");
        if (!std::filesystem::is_directory(config_.directory))
            throw std::runtime_error("Benchmark directory does not exist");
        if (config_.terrain == "benchmark_stairs") {
            const int body = mj_name2id(m, mjOBJ_BODY, "benchmark_stairs");
            if (body < 0) throw std::runtime_error("Benchmark stair body is missing");
            course_center_y_ = d->xpos[3 * body + 1];
            for (const char *foot : {"FL", "FR", "RL", "RR"}) {
                const int id = mj_name2id(m, mjOBJ_GEOM, foot);
                if (id < 0 || m->geom_priority[id] != 1 ||
                    std::abs(m->geom_friction[3 * id] - 0.4) > 1e-6)
                    throw std::runtime_error("Expected Go2 foot friction 0.4 and priority 1");
            }
            for (int step = 1; step <= 11; ++step) {
                const std::string name = step == 11 ? "benchmark_stair_landing" :
                    "benchmark_step" + std::to_string(step);
                const int id = mj_name2id(m, mjOBJ_GEOM, name.c_str());
                if (id < 0) throw std::runtime_error("Benchmark stair geom is missing: " + name);
                if (config_.stair_contact_friction > 0.4) {
                    // Equal foot/stair priority preserves six-dimensional contacts;
                    // MuJoCo then takes the larger sliding friction coefficient.
                    m->geom_priority[id] = 1;
                    m->geom_friction[3 * id] = config_.stair_contact_friction;
                }
            }
            std::cout << "[VelocityTrial] effective foot-stair sliding friction="
                      << config_.stair_contact_friction << std::endl;
        }
        last_sim_time_ = d->time;
        initialized_ = true;
    }
    static constexpr double kDeg = 180.0 / 3.14159265358979323846;
    Config config_;
    std::ofstream csv_;
    std::unique_ptr<BoxAscent> ascent_;
    int base_id_ = -1;
    bool initialized_ = false, armed_ = false, finished_ = false;
    long long wall0_ns_ = 0;
    double last_sim_time_ = 0, next_status_ = 0, next_sample_ = 0;
    double t0_ = 0, x0_ = 0, y0_ = 0, last_x_ = 0, last_sample_ = 0;
    double goal_x_ = 0, goal_time_ = -1, error_integral_ = 0, max_tilt_ = 0, fall_duration_ = 0;
    double course_center_y_ = 0;
};
} // namespace velocity_benchmark
