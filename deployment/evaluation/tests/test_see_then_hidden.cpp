#include "../../unitree_mujoco/simulate/src/see_then_hidden.h"
#include "../see_then_hidden_protocol.h"

#ifdef NDEBUG
#undef NDEBUG  // Keep the checks active in the documented Release build.
#endif
#include <cassert>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <vector>

int main() {
    const auto scene = std::filesystem::temp_directory_path() / "see_then_hidden_test.xml";
    {
        std::ofstream file(scene);
        file << R"(<mujoco><worldbody><body pos="1.7 0 0.4"><geom name="box1"
                 type="box" size="0.2 0.3 0.3"/></body></worldbody></mujoco>)";
    }
    char error[1024]{};
    mjModel *model = mj_loadXML(scene.c_str(), nullptr, error, sizeof(error));
    assert(model && error[0] == '\0');
    mjData *data = mj_makeData(model);
    mj_forward(model, data);
    const int target = mj_name2id(model, mjOBJ_GEOM, "box1");
    assert(target >= 0);

    SeeThenHidden experiment(1.2, 3, "box1", 60, "");
    const mjtNum identity[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    const int hits[1] = {target};
    const mjtNum distances[1] = {1.5};
    std::vector<uint8_t> mask;

    // Being near the box without a LiDAR hit must never satisfy the preview.
    const int missed[1] = {-1};
    const mjtNum near_sensor[3] = {0.4, 0, 0.4};
    data->time = 0.0;
    assert(!experiment.update(model, data, near_sensor, identity, missed,
                              distances, 1, &mask));
    assert(mask[2 * 60 + 30] == 1); // preview mask is metadata, not yet applied

    const mjtNum far_sensor[3] = {0, 0, 0.4};
    for (int scan = 0; scan < 3; ++scan) {
        data->time = 0.05 * (scan + 1);
        assert(!experiment.update(model, data, far_sensor, identity, hits,
                                  distances, 1, &mask));
    }
    data->time = 0.20;
    assert(experiment.update(model, data, near_sensor, identity, hits,
                             distances, 1, &mask));
    assert(mask.size() == 25U * 60U);
    assert(mask[2 * 60 + 30] == 1);
    assert(mask[2 * 60 + 0] == 0);

    // The extended wire message must preserve the legacy points and expose
    // the mask only when the matching FOV is configured.
    constexpr size_t pose_bytes = 12 * sizeof(float);
    constexpr size_t point_bytes_expected = 3 * sizeof(float);
    const see_then_hidden_wire::MaskFooter footer{
        see_then_hidden_wire::kMagic, 25, 60, 1};
    std::vector<uint8_t> message(pose_bytes + point_bytes_expected + mask.size() +
                                 sizeof(footer), 0);
    std::memcpy(message.data() + pose_bytes + point_bytes_expected,
                mask.data(), mask.size());
    std::memcpy(message.data() + message.size() - sizeof(footer), &footer,
                sizeof(footer));
    size_t parsed_points = 0;
    const uint8_t *parsed_mask = nullptr;
    bool active = false;
    using see_then_hidden_wire::ParseResult;
    std::vector<uint8_t> legacy(pose_bytes + point_bytes_expected, 0);
    assert(see_then_hidden_wire::parseMaskSuffix(
               legacy.data(), legacy.size(), pose_bytes, 25, 60,
               &parsed_points, &parsed_mask, &active) == ParseResult::kLegacy);
    assert(see_then_hidden_wire::parseMaskSuffix(
               message.data(), message.size(), pose_bytes, 25, 60,
               &parsed_points, &parsed_mask, &active) == ParseResult::kValid);
    assert(parsed_points == point_bytes_expected && active && parsed_mask[2 * 60 + 30]);
    assert(see_then_hidden_wire::parseMaskSuffix(
               message.data(), message.size(), pose_bytes, 25, 120,
               &parsed_points, &parsed_mask, &active) == ParseResult::kInvalid);

    data->time = 0.0;
    assert(!experiment.update(model, data, near_sensor, identity, hits,
                              distances, 1, &mask));
    mj_deleteData(data);
    mj_deleteModel(model);
    std::filesystem::remove(scene);
}
