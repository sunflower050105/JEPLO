#pragma once

#include <mujoco/mujoco.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

// Simulation-only intervention. The policy receives the normal LiDAR stream
// until the named box has been hit by several distinct scans and the sensor is
// within trigger_distance_m of its surface. Thereafter a projected box-region
// mask is sent to the depth publisher on every scan until a simulation reset.
class SeeThenHidden {
  public:
    SeeThenHidden(double trigger_distance_m, int preview_scans, std::string target_geom,
                  int columns, const std::string &log_path)
        : trigger_distance_m_(trigger_distance_m)
        , preview_scans_(preview_scans)
        , target_geom_(std::move(target_geom))
        , columns_(columns) {
        if (!log_path.empty()) {
            log_.open(log_path);
            if (!log_) {
                std::cerr << "[SeeThenHidden] Cannot open CSV log: " << log_path << '\n';
            } else {
                log_ << "sim_time_s,target_range_m,target_hits,consecutive_preview_scans,"
                        "hidden,mask_pixels,sensor_x,sensor_y,sensor_z,target_x,target_y,target_z\n";
                log_.flush();
            }
        }
    }

    bool update(const mjModel *model, const mjData *data, const mjtNum *sensor_position,
                const mjtNum *sensor_rotation, const int *hit_geom_ids,
                const mjtNum *hit_distances, int ray_count, std::vector<uint8_t> *mask) {
        mask->assign(static_cast<size_t>(kRows * columns_), 0);
        if (data->time < previous_sim_time_) {
            reset("simulation time reset");
        }
        previous_sim_time_ = data->time;

        if (model != model_) {
            model_ = model;
            geom_id_ = mj_name2id(model, mjOBJ_GEOM, target_geom_.c_str());
            if (geom_id_ < 0 || model->geom_type[geom_id_] != mjGEOM_BOX) {
                std::cerr << "[SeeThenHidden] Target must be a box geom: " << target_geom_
                          << '\n';
                return false;
            }
            reset("model loaded");
        }

        const mjtNum *center = data->geom_xpos + 3 * geom_id_;
        const mjtNum *rotation = data->geom_xmat + 9 * geom_id_;
        const mjtNum *half_size = model->geom_size + 3 * geom_id_;
        const double range = distanceToBox(sensor_position, center, rotation, half_size);
        if (range > 10.0) {
            if (hidden_ || preview_count_ > 0) {
                reset("target moved out of scene");
            }
            return false;
        }

        int hits = 0;
        for (int i = 0; i < ray_count; ++i) {
            if (hit_geom_ids[i] == geom_id_ && hit_distances[i] >= 0.1 &&
                hit_distances[i] <= 2.0) {
                ++hits;
            }
        }
        if (!hidden_) {
            preview_count_ = hits > 0 ? preview_count_ + 1 : 0;
            if (preview_count_ >= preview_scans_ && range <= trigger_distance_m_) {
                hidden_ = true;
                std::cout << "[SeeThenHidden] HIDE at simulation time " << std::fixed
                          << std::setprecision(3) << data->time << " s, box range " << range
                          << " m, after " << preview_count_ << " visible scans\n";
            }
        }

        // Send the projected region during the preview too. The publisher
        // stores it alongside each clean scan, so it can remove only the old
        // box pixels when hiding starts, leaving other old depth intact.
        const size_t projected_pixels = projectBoxMask(sensor_position, sensor_rotation,
                                                        center, rotation, half_size, mask);
        const size_t masked_pixels = hidden_ ? projected_pixels : 0;
        if (log_) {
            log_ << std::fixed << std::setprecision(4) << data->time << ',' << range << ','
                 << hits << ',' << preview_count_ << ',' << static_cast<int>(hidden_)
                 << ',' << masked_pixels;
            for (int axis = 0; axis < 3; ++axis) log_ << ',' << sensor_position[axis];
            for (int axis = 0; axis < 3; ++axis) log_ << ',' << center[axis];
            log_ << '\n';
            log_.flush();
        }
        return hidden_;
    }

  private:
    static constexpr int kRows = 25;
    static constexpr double kPi = 3.14159265358979323846;
    static constexpr double kDegreesToRadians = kPi / 180.0;

    void reset(const char *reason) {
        if (hidden_ || preview_count_ > 0) {
            std::cout << "[SeeThenHidden] Reset: " << reason << '\n';
        }
        hidden_ = false;
        preview_count_ = 0;
    }

    static double distanceToBox(const mjtNum *point, const mjtNum *center,
                                const mjtNum *rotation, const mjtNum *half_size) {
        mjtNum delta[3], local[3];
        mju_sub3(delta, point, center);
        mju_mulMatTVec(local, rotation, delta, 3, 3);
        double squared = 0.0;
        for (int axis = 0; axis < 3; ++axis) {
            const double outside = std::max(0.0, std::abs(local[axis]) - half_size[axis]);
            squared += outside * outside;
        }
        return std::sqrt(squared);
    }

    size_t projectBoxMask(const mjtNum *sensor_position, const mjtNum *sensor_rotation,
                          const mjtNum *center, const mjtNum *box_rotation,
                          const mjtNum *half_size, std::vector<uint8_t> *mask) const {
        double min_row = 1e9, max_row = -1e9, min_col = 1e9, max_col = -1e9;
        int visible_corners = 0;
        for (int corner = 0; corner < 8; ++corner) {
            mjtNum offset[3] = {
                (corner & 1 ? 1.0 : -1.0) * half_size[0],
                (corner & 2 ? 1.0 : -1.0) * half_size[1],
                (corner & 4 ? 1.0 : -1.0) * half_size[2],
            };
            mjtNum world_offset[3], world[3], delta[3], local[3];
            mju_mulMatVec3(world_offset, box_rotation, offset);
            mju_add3(world, center, world_offset);
            mju_sub3(delta, world, sensor_position);
            mju_mulMatTVec(local, sensor_rotation, delta, 3, 3);
            if (local[0] <= 0.05) continue;

            const double radius = mju_norm3(local);
            const double elevation = std::asin(std::clamp(local[2] / radius, -1.0, 1.0));
            const double azimuth = std::atan2(local[1], local[0]);
            const double row = (elevation + 4.5 * kDegreesToRadians) /
                               (2.0 * kDegreesToRadians);
            const double col = (azimuth + static_cast<double>(columns_) * kDegreesToRadians) /
                               (2.0 * kDegreesToRadians);
            min_row = std::min(min_row, row);
            max_row = std::max(max_row, row);
            min_col = std::min(min_col, col);
            max_col = std::max(max_col, col);
            ++visible_corners;
        }
        if (visible_corners == 0) return 0;

        // One-pixel margin accounts for the publisher's round-to-nearest binning.
        const int row0 = std::max(0, static_cast<int>(std::floor(min_row)) - 1);
        const int row1 = std::min(kRows - 1, static_cast<int>(std::ceil(max_row)) + 1);
        const int col0 = std::max(0, static_cast<int>(std::floor(min_col)) - 1);
        const int col1 = std::min(columns_ - 1, static_cast<int>(std::ceil(max_col)) + 1);
        if (row0 > row1 || col0 > col1) return 0;
        for (int row = row0; row <= row1; ++row) {
            for (int col = col0; col <= col1; ++col) {
                (*mask)[static_cast<size_t>(row) * columns_ + col] = 1;
            }
        }
        return static_cast<size_t>(row1 - row0 + 1) * (col1 - col0 + 1);
    }

    double trigger_distance_m_;
    int preview_scans_;
    std::string target_geom_;
    int columns_;
    std::ofstream log_;
    const mjModel *model_ = nullptr;
    int geom_id_ = -1;
    double previous_sim_time_ = -1.0;
    int preview_count_ = 0;
    bool hidden_ = false;
};
