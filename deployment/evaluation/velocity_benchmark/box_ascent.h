#pragma once

#include <mujoco/mujoco.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>

namespace velocity_benchmark {

// Evaluation only: no signal from this detector is supplied to the controller.
// Contact means geometric foot/top contact, not a measured load-bearing force.
class BoxAscent {
  public:
    static constexpr double kContactWindow = 0.5;
    static constexpr double kHold = 0.2;
    static constexpr double kMaxSampleGap = 0.1;
    static constexpr double kMaxTiltDeg = 30.0;
    static constexpr double kMinClearance = 0.16;
    static constexpr double kTopTolerance = 0.03;
    static constexpr double kContactDistance = 0.005;
    static constexpr double kMinNormalAlignment = 0.7;
    static constexpr unsigned kAllFeet = 15;  // bits: FL, FR, RL, RR

    struct State {
        unsigned on_top_mask = 0;
        unsigned contact_mask = 0;
        unsigned recent_contact_mask = 0;
        bool candidate = false;
        double hold_s = 0;
        double achieved_at = -1;
    };

    explicit BoxAscent(const mjModel *m)
        : data_(mj_makeData(m), mj_deleteData) {
        if (!data_) throw std::runtime_error("Cannot allocate ascent scoring state");
        box_ = mj_name2id(m, mjOBJ_GEOM, "box2");
        base_ = mj_name2id(m, mjOBJ_BODY, "base_link");
        if (box_ < 0 || base_ < 0 || m->geom_type[box_] != mjGEOM_BOX)
            throw std::runtime_error("Box ascent requires base_link and box2 box geometry");
        const char *names[] = {"FL", "FR", "RL", "RR"};
        for (int i = 0; i < 4; ++i) {
            feet_[i] = mj_name2id(m, mjOBJ_GEOM, names[i]);
            if (feet_[i] < 0 || m->geom_type[feet_[i]] != mjGEOM_SPHERE)
                throw std::runtime_error("Box ascent requires FL/FR/RL/RR foot spheres");
        }
        last_contact_.fill(-std::numeric_limits<double>::infinity());
    }

    const State &update(const mjModel *m, const mjData *source, double elapsed) {
        if (elapsed < last_sample_)
            throw std::runtime_error("Ascent scoring time must be monotonic");
        if (last_sample_ >= 0 && elapsed - last_sample_ > kMaxSampleGap + 1e-9) {
            candidate_since_ = -1;
            last_contact_.fill(-std::numeric_limits<double>::infinity());
        }
        last_sample_ = elapsed;

        // Refresh geometry from the exact qpos saved in state.csv without
        // changing the live simulation or integrating a second physics step.
        mju_copy(data_->qpos, source->qpos, m->nq);
        if (m->nmocap) {
            mju_copy(data_->mocap_pos, source->mocap_pos, 3 * m->nmocap);
            mju_copy(data_->mocap_quat, source->mocap_quat, 4 * m->nmocap);
        }
        mj_fwdPosition(m, data_.get());
        const double *rotation = data_->geom_xmat + 9 * box_;
        const double *size = m->geom_size + 3 * box_;
        state_.on_top_mask = state_.contact_mask = state_.recent_contact_mask = 0;
        for (int i = 0; i < 4; ++i) {
            const auto p = local(data_->geom_xpos + 3 * feet_[i]);
            const double radius = m->geom_size[3 * feet_[i]];
            if (std::abs(p[0]) + radius <= size[0] &&
                std::abs(p[1]) + radius <= size[1] &&
                p[2] - radius >= size[2] - kTopTolerance)
                state_.on_top_mask |= 1u << i;
        }
        for (int j = 0; j < data_->ncon; ++j) {
            const mjContact &c = data_->contact[j];
            const int other = c.geom[0] == box_ ? c.geom[1] :
                              c.geom[1] == box_ ? c.geom[0] : -1;
            if (other < 0 || c.dist > kContactDistance) continue;
            const auto p = local(c.pos);
            // MuJoCo's first contact-frame axis is the normal. Its sign depends
            // on geom ordering; require alignment with the top normal either way.
            const double alignment = std::abs(c.frame[0] * rotation[2] +
                c.frame[1] * rotation[5] + c.frame[2] * rotation[8]);
            if (std::abs(p[2] - size[2]) > kTopTolerance ||
                alignment < kMinNormalAlignment) continue;
            for (int i = 0; i < 4; ++i) {
                if (other == feet_[i]) {
                    state_.contact_mask |= 1u << i;
                    last_contact_[i] = elapsed;
                }
            }
        }
        for (int i = 0; i < 4; ++i)
            if (elapsed - last_contact_[i] <= kContactWindow + 1e-9)
                state_.recent_contact_mask |= 1u << i;

        const auto base = local(data_->xpos + 3 * base_);
        const double tilt = std::acos(std::clamp(data_->xmat[9 * base_ + 8], -1.0, 1.0)) *
                            180.0 / 3.14159265358979323846;
        state_.candidate = state_.on_top_mask == kAllFeet &&
            state_.recent_contact_mask == kAllFeet &&
            std::abs(base[0]) <= size[0] && std::abs(base[1]) <= size[1] &&
            base[2] - size[2] >= kMinClearance && tilt <= kMaxTiltDeg;
        if (!state_.candidate) candidate_since_ = -1;
        else if (candidate_since_ < 0) candidate_since_ = elapsed;
        state_.hold_s = state_.candidate ? elapsed - candidate_since_ : 0;
        if (state_.achieved_at < 0 && state_.candidate && state_.hold_s + 1e-9 >= kHold)
            state_.achieved_at = elapsed;
        return state_;
    }

  private:
    std::array<double, 3> local(const double *world) const {
        const double *origin = data_->geom_xpos + 3 * box_;
        const double *r = data_->geom_xmat + 9 * box_;
        std::array<double, 3> p{};
        for (int axis = 0; axis < 3; ++axis)
            for (int j = 0; j < 3; ++j) p[axis] += r[3 * j + axis] * (world[j] - origin[j]);
        return p;
    }
    std::unique_ptr<mjData, decltype(&mj_deleteData)> data_;
    int box_ = -1, base_ = -1;
    std::array<int, 4> feet_{};
    std::array<double, 4> last_contact_{};
    double candidate_since_ = -1, last_sample_ = -1;
    State state_;
};
} // namespace velocity_benchmark
