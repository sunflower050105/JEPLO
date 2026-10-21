#pragma once

#include <algorithm>
#include <cmath>

// Heading reference is captured from the IMU on the first forward command.
// This helper changes the policy's requested yaw rate, never the forward speed.
class ImuYawHold {
  public:
    struct Output {
        bool valid = true;
        bool active = false;
        float target_yaw = 0.0f;
        float measured_yaw = 0.0f;
        float error = 0.0f;
        float wz = 0.0f;
    };

    void reset() { armed_ = false; }

    Output update(float qw, float qx, float qy, float qz, float gyro_z,
                  float forward_command, float kp, float kd, float max_wz) {
        Output out;
        if (forward_command <= 0.05f) {
            reset();
            return out;
        }
        const float norm_sq = qw * qw + qx * qx + qy * qy + qz * qz;
        if (!std::isfinite(norm_sq) || norm_sq < 1e-6f || !std::isfinite(gyro_z)) {
            out.valid = false;
            return out;
        }
        const float inv_norm = 1.0f / std::sqrt(norm_sq);
        qw *= inv_norm;
        qx *= inv_norm;
        qy *= inv_norm;
        qz *= inv_norm;
        out.measured_yaw = std::atan2(2.0f * (qw * qz + qx * qy),
                                      1.0f - 2.0f * (qy * qy + qz * qz));
        if (!armed_) {
            target_yaw_ = out.measured_yaw;
            armed_ = true;
        }
        out.active = true;
        out.target_yaw = target_yaw_;
        out.error = std::atan2(std::sin(target_yaw_ - out.measured_yaw),
                               std::cos(target_yaw_ - out.measured_yaw));
        out.wz = std::clamp(kp * out.error - kd * gyro_z, -max_wz, max_wz);
        return out;
    }

  private:
    bool armed_ = false;
    float target_yaw_ = 0.0f;
};
