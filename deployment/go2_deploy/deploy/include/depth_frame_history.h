#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

// A completed depth image and the timestamps carried with it. Sequence 0 is
// startup padding; positive sequences identify distinct received images.
struct DepthHistoryFrame {
    std::vector<float> pixels;
    uint64_t scan_ns = 0;
    uint64_t publish_ns = 0;
    uint64_t recv_ns = 0;
    uint64_t seq = 0;
};

// Stores [older, newer, newest]. A control tick only reads the history; every
// accepted image advances it once, even if several arrive between control
// ticks. Delayed mode selects [older, newer], matching Isaac's depth_delay=1;
// latest-two mode selects [newer, newest]. Distinct IDs do not guarantee
// different pixel contents.
class DepthFrameHistory {
  public:
    explicit DepthFrameHistory(std::size_t pixels = 0) { reset(pixels); }

    void reset(std::size_t pixels) {
        pixels_ = pixels;
        for (auto &frame : frames_) {
            frame = {};
            frame.pixels.assign(pixels, 0.0f);
        }
    }

    void push(const float *pixels, uint64_t seq, uint64_t scan_ns,
              uint64_t publish_ns, uint64_t recv_ns) {
        frames_[0] = std::move(frames_[1]);
        frames_[1] = std::move(frames_[2]);
        auto &newest = frames_[2];
        newest.pixels.assign(pixels, pixels + pixels_);
        newest.seq = seq;
        newest.scan_ns = scan_ns;
        newest.publish_ns = publish_ns;
        newest.recv_ns = recv_ns;
    }

    const DepthHistoryFrame &older() const { return frames_[0]; }
    const DepthHistoryFrame &newer() const { return frames_[1]; }
    const DepthHistoryFrame &newest() const { return frames_[2]; }

  private:
    std::size_t pixels_ = 0;
    std::array<DepthHistoryFrame, 3> frames_;
};
