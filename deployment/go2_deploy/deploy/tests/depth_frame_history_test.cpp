#include "depth_frame_history.h"

#include <array>
#include <stdexcept>

void require(bool condition) {
    if (!condition) throw std::runtime_error("depth frame history mismatch");
}

int main() {
    DepthFrameHistory history(2);
    auto expect = [&](uint64_t older, uint64_t newer, float older_pixel,
                      float newer_pixel) {
        require(history.older().seq == older);
        require(history.newer().seq == newer);
        require(history.older().pixels == std::vector<float>({older_pixel, older_pixel}));
        require(history.newer().pixels == std::vector<float>({newer_pixel, newer_pixel}));
    };
    auto expect_latest_two = [&](uint64_t previous, uint64_t latest,
                                 float previous_pixel, float latest_pixel) {
        require(history.newer().seq == previous);
        require(history.newest().seq == latest);
        require(history.newer().pixels == std::vector<float>({previous_pixel, previous_pixel}));
        require(history.newest().pixels == std::vector<float>({latest_pixel, latest_pixel}));
    };

    expect(0, 0, 0.0f, 0.0f);
    expect_latest_two(0, 0, 0.0f, 0.0f);
    const std::array<float, 2> first{1.0f, 1.0f};
    const std::array<float, 2> second{2.0f, 2.0f};
    const std::array<float, 2> third{3.0f, 3.0f};
    const std::array<float, 2> fourth{4.0f, 4.0f};
    history.push(first.data(), 1, 100, 110, 120);
    expect(0, 0, 0.0f, 0.0f);
    expect_latest_two(0, 1, 0.0f, 1.0f);
    history.push(second.data(), 2, 200, 210, 220);
    expect(0, 1, 0.0f, 1.0f);
    expect_latest_two(1, 2, 1.0f, 2.0f);
    history.push(third.data(), 3, 300, 310, 320);
    expect(1, 2, 1.0f, 2.0f);
    expect_latest_two(2, 3, 2.0f, 3.0f);
    require(history.newer().scan_ns == 200);
    require(history.newer().publish_ns == 210);
    require(history.newer().recv_ns == 220);

    // Multiple 50 Hz control reads without a 10 Hz image leave the pair alone.
    expect(1, 2, 1.0f, 2.0f);
    expect(1, 2, 1.0f, 2.0f);
    expect_latest_two(2, 3, 2.0f, 3.0f);
    history.push(fourth.data(), 4, 400, 410, 420);
    expect(2, 3, 2.0f, 3.0f);
    expect_latest_two(3, 4, 3.0f, 4.0f);
    require(history.newest().scan_ns == 400);
    require(history.newest().publish_ns == 410);
    require(history.newest().recv_ns == 420);

    history.reset(2);
    expect(0, 0, 0.0f, 0.0f);
    expect_latest_two(0, 0, 0.0f, 0.0f);
}
