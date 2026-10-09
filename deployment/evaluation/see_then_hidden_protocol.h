#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

// Optional suffix on the simulator's existing point-cloud message:
// [12 float pose][N * 3 float points][rows * cols uint8 target mask][MaskFooter].
// A legacy message has no suffix. The depth publisher applies the mask after
// temporal min aggregation so old scans cannot reveal the hidden target.
namespace see_then_hidden_wire {

constexpr uint32_t kMagic = 0x31544853; // "SHT1" in little-endian byte order

struct MaskFooter {
    uint32_t magic;
    uint32_t rows;
    uint32_t cols;
    uint32_t active;
};
static_assert(sizeof(MaskFooter) == 4 * sizeof(uint32_t));

enum class ParseResult { kLegacy, kValid, kInvalid };

inline ParseResult parseMaskSuffix(
    const uint8_t *message, size_t size, size_t pose_bytes, uint32_t expected_rows,
    uint32_t expected_cols, size_t *point_bytes, const uint8_t **mask, bool *active) {
    *mask = nullptr;
    *active = false;
    if (size < pose_bytes) {
        return ParseResult::kInvalid;
    }
    if (size >= pose_bytes + sizeof(MaskFooter)) {
        MaskFooter footer{};
        std::memcpy(&footer, message + size - sizeof(footer), sizeof(footer));
        if (footer.magic == kMagic) {
            const size_t mask_bytes = static_cast<size_t>(expected_rows) * expected_cols;
            if (footer.rows != expected_rows || footer.cols != expected_cols ||
                footer.active > 1 || size < pose_bytes + mask_bytes + sizeof(footer)) {
                return ParseResult::kInvalid;
            }
            *point_bytes = size - pose_bytes - mask_bytes - sizeof(footer);
            if (*point_bytes % (3 * sizeof(float)) != 0) {
                return ParseResult::kInvalid;
            }
            *mask = message + pose_bytes + *point_bytes;
            *active = footer.active == 1;
            return ParseResult::kValid;
        }
    }
    *point_bytes = size - pose_bytes;
    return *point_bytes % (3 * sizeof(float)) == 0 ? ParseResult::kLegacy
                                                  : ParseResult::kInvalid;
}

} // namespace see_then_hidden_wire
