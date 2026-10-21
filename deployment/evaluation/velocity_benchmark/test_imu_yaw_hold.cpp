#include "../../go2_deploy/deploy/src/imu_yaw_hold.h"

#include <cassert>
#include <cmath>

static ImuYawHold::Output at_yaw(ImuYawHold &hold, float yaw, float forward = 0.8f,
                                  float gyro_z = 0.0f) {
    return hold.update(std::cos(yaw / 2), 0, 0, std::sin(yaw / 2), gyro_z,
                       forward, 2.0f, 0.25f, 1.0f);
}

int main() {
    ImuYawHold hold;
    auto start = at_yaw(hold, 0.4f);
    assert(start.active && std::abs(start.error) < 1e-5f);
    auto turned = at_yaw(hold, 0.7f);
    assert(std::abs(turned.error + 0.3f) < 1e-5f);
    assert(std::abs(turned.wz + 0.6f) < 1e-5f);
    auto damping = at_yaw(hold, 0.7f, 0.8f, 0.4f);
    assert(damping.wz < turned.wz);
    assert(at_yaw(hold, 1.2f).wz == -1.0f); // bounded correction
    assert(!at_yaw(hold, 1.2f, 0).active);   // stop releases the reference
    assert(std::abs(at_yaw(hold, -0.6f).error) < 1e-5f); // next start relatches

    ImuYawHold wrap;
    const float pi = std::acos(-1.0f);
    at_yaw(wrap, pi - 0.02f);
    auto across_wrap = at_yaw(wrap, -pi + 0.02f);
    assert(std::abs(across_wrap.error + 0.04f) < 1e-4f);
    auto bad = wrap.update(0, 0, 0, 0, 0, 0.8f, 2.0f, 0.25f, 1.0f);
    assert(!bad.valid);
    assert(std::abs(at_yaw(wrap, -pi + 0.02f).error + 0.04f) < 1e-4f);
}
