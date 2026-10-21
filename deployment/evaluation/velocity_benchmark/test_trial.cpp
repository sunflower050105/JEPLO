#include "velocity_trial.h"
#include <functional>
#include <sstream>
#include <vector>

static void require(bool condition, const std::string &message) {
    if (!condition) throw std::runtime_error(message);
}

int main() {
    const auto root = std::filesystem::temp_directory_path() /
        ("jeplo_velocity_test_" + std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(root);
    const auto xml = root / "scene.xml";
    std::ofstream(xml) << R"(<mujoco><worldbody>
      <geom type="plane" size="10 10 .1"/>
      <body name="base_link" pos="0 0 .3"><freejoint/>
        <geom type="box" size=".2 .1 .05" mass="10"/>
        <body pos=".18 .1 -.28"><joint type="slide" axis="0 0 1"/>
          <geom name="FL" type="sphere" size=".02"/></body>
        <body pos=".18 -.1 -.28"><joint type="slide" axis="0 0 1"/>
          <geom name="FR" type="sphere" size=".02"/></body>
        <body pos="-.18 .1 -.28"><joint type="slide" axis="0 0 1"/>
          <geom name="RL" type="sphere" size=".02"/></body>
        <body pos="-.18 -.1 -.28"><joint type="slide" axis="0 0 1"/>
          <geom name="RR" type="sphere" size=".02"/></body>
      </body>
      <geom name="box2" type="box" pos="2 0 .2" size=".5 .75 .2"/>
      <geom name="benchmark_stair_landing" type="box" pos="3.6 0 1" size=".6 .8 1"/>
    </worldbody></mujoco>)";
    char error[1024];
    mjModel *model = mj_loadXML(xml.c_str(), nullptr, error, sizeof(error));
    require(model, error);
    const auto run = [&](const std::string &name, const std::string &terrain,
                         double duration, const std::function<void(mjData *, double)> &pose,
                         const std::string &outcome, const std::string &box_task = "ascent") {
        const auto directory = root / name;
        std::filesystem::create_directory(directory);
        std::ofstream(directory / "GO");
        mjData *data = mj_makeData(model);
        velocity_benchmark::Trial trial({directory.string(), terrain, duration, 1.0, 0.8, box_task});
        bool done = false;
        for (int i = 0; i <= static_cast<int>((duration + 0.5) * 100) && !done; ++i) {
            const double t = i / 100.0;
            data->time = t;
            pose(data, t);
            mj_forward(model, data);
            done = trial.update(model, data);
        }
        require(done, name + " did not finish");
        std::ifstream result(directory / "result.json");
        std::ostringstream content;
        content << result.rdbuf();
        require(content.str().find("\"outcome\": \"" + outcome + "\"") != std::string::npos,
                name + " wrong outcome: " + content.str());
        require(std::filesystem::file_size(directory / "scene.mjb") > 0, "Missing replay model");
        if (terrain == "boxes" && box_task == "ascent") {
            require(content.str().find("\"scoring_rule\": \"box_ascent_v2\"") != std::string::npos,
                    "Missing ascent scoring version");
            require(content.str().find("\"goal_x_m\": null") != std::string::npos,
                    "Ascent must not report a misleading X finish line");
        }
        mj_deleteData(data);
    };
    run("moving", "flat", 1.0, [](mjData *d, double t) { d->qpos[0] = t; }, "success");
    run("stationary", "flat", 1.0, [](mjData *, double) {}, "timeout");
    run("fall", "flat", 1.0, [](mjData *d, double) { d->qpos[2] = 0.1; }, "fall");
    run("transient", "flat", 1.0, [](mjData *d, double t) {
        d->qpos[0] = t; d->qpos[2] = t < 0.1 ? 0.1 : 0.3;
    }, "success");
    run("bypass", "boxes", 4.0, [](mjData *d, double t) {
        d->qpos[0] = 2 * t; d->qpos[1] = t;
    }, "left_course");
    run("boxes", "boxes", 5.0, [](mjData *d, double t) { d->qpos[0] = t; }, "success", "traverse");
    run("finish_then_fall", "boxes", 5.0, [](mjData *d, double t) {
        d->qpos[0] = t; d->qpos[2] = t > 3.0 ? 0.1 : 0.3;
    }, "fall", "traverse");
    run("ascent_before_descent", "boxes", 2.0, [](mjData *d, double t) {
        if (t >= 0.2) { d->qpos[0] = 2; d->qpos[2] = t < 0.8 ? 0.699 : 0.1; }
    }, "success");
    run("front_feet_only", "boxes", 0.6, [](mjData *d, double) {
        d->qpos[0] = 1.4; d->qpos[2] = 0.699;
        d->qpos[9] = d->qpos[10] = -0.4;
    }, "timeout");
    run("flying_over_top", "boxes", 0.6, [](mjData *d, double) {
        d->qpos[0] = 2; d->qpos[2] = 0.9;
    }, "timeout");
    run("brief_touch", "boxes", 0.8, [](mjData *d, double t) {
        d->qpos[0] = t >= 0.2 && t < 0.3 ? 2 : 0; d->qpos[2] = 0.699;
    }, "timeout");
    run("ascent_after_deadline", "boxes", 0.51, [](mjData *d, double t) {
        if (t >= 0.32) { d->qpos[0] = 2; d->qpos[2] = 0.699; }
    }, "timeout");
    run("stair_landing", "benchmark_stairs", 1.0, [](mjData *d, double t) {
        if (t >= 0.2) { d->qpos[0] = 3.6; d->qpos[2] = 2.299; }
    }, "success");
    run("stair_last_tread_only", "benchmark_stairs", 0.8, [](mjData *d, double t) {
        if (t >= 0.2) { d->qpos[0] = 2.85; d->qpos[2] = 2.299; }
    }, "timeout");

    // Geometric and temporal checks independent of the trial terminal conditions.
    using velocity_benchmark::BoxAscent;
    mjData *data = mj_makeData(model);
    data->qpos[0] = 2; data->qpos[2] = 0.699;
    const auto reset_pose = [&]() {
        mj_resetData(model, data);
        data->qpos[0] = 2; data->qpos[2] = 0.699;
    };
    BoxAscent trot(model);
    for (int i = 0; i <= 20; ++i) {
        const bool diagonal = (i / 5) % 2;
        data->qpos[7] = data->qpos[10] = diagonal ? 0.05 : 0;
        data->qpos[8] = data->qpos[9] = diagonal ? 0 : 0.05;
        const auto s = trot.update(model, data, i * 0.02);
        require(s.contact_mask == (diagonal ? 6u : 9u), "Unexpected diagonal contacts");
    }
    require(trot.update(model, data, 0.42).achieved_at >= 0, "Trotting ascent rejected");
    data->qpos[2] = 0.1;
    require(trot.update(model, data, 0.44).achieved_at >= 0, "Later descent revoked ascent");

    reset_pose();
    BoxAscent interruption(model);
    require(interruption.update(model, data, 0).candidate, "Four feet on top not detected");
    interruption.update(model, data, 0.08);
    data->qpos[0] = 1.4;
    require(!interruption.update(model, data, 0.1).candidate, "Leaving footprint did not reset hold");
    data->qpos[0] = 2;
    interruption.update(model, data, 0.12);
    interruption.update(model, data, 0.2);
    require(interruption.update(model, data, 0.28).achieved_at < 0, "Interrupted holds accumulated");
    require(interruption.update(model, data, 0.32).achieved_at >= 0, "Continuous hold rejected");

    reset_pose();
    BoxAscent gap(model);
    gap.update(model, data, 0);
    require(gap.update(model, data, 0.3).achieved_at < 0, "Missing samples counted as continuous hold");

    reset_pose();
    BoxAscent expired(model);
    expired.update(model, data, 0);
    data->qpos[0] = 1.4;  // Break hold while old contacts age out.
    data->qpos[2] = 0.9;
    for (int i = 1; i <= 30; ++i) expired.update(model, data, i * 0.02);
    data->qpos[0] = 2;
    const auto stale = expired.update(model, data, 0.62);
    require(stale.on_top_mask == 15 && stale.recent_contact_mask == 0 && !stale.candidate,
            "Expired contacts allowed an unsupported ascent");

    reset_pose();
    BoxAscent side(model);
    data->qpos[0] = 1.301; data->qpos[2] = 0.58;
    require(side.update(model, data, 0).contact_mask == 0, "Riser contacts counted as top contacts");
    reset_pose();
    BoxAscent belly(model);
    data->qpos[2] = 0.5;
    for (int i = 7; i < 11; ++i) data->qpos[i] = 0.199;
    const auto low = belly.update(model, data, 0);
    require(low.contact_mask == 15 && !low.candidate, "Belly on platform counted as upright ascent");
    reset_pose();
    BoxAscent leaning(model);
    const double pitch = 40.0 * 3.14159265358979323846 / 180.0;
    data->qpos[3] = std::cos(pitch / 2); data->qpos[5] = std::sin(pitch / 2);
    for (int i = 0; i < 4; ++i) {
        const double foot_x = i < 2 ? 0.18 : -0.18;
        // Keep each foot in contact while pitching the torso past the limit.
        data->qpos[7 + i] = (0.419 - data->qpos[2] + std::sin(pitch) * foot_x) /
                           std::cos(pitch) + 0.28;
    }
    const auto tilted = leaning.update(model, data, 0);
    require(tilted.contact_mask == 15 && tilted.on_top_mask == 15 && !tilted.candidate,
            "Excessive torso tilt counted as upright ascent");
    mj_deleteData(data);
    mj_deleteModel(model);
    std::filesystem::remove_all(root);
    std::cout << "Velocity trial scoring tests passed\n";
}
