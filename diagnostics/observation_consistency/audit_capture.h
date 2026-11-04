// Used ONLY by the generated diagnostic controller. No baseline includes this.
// Records native model tensors, without changing their contents or scheduling.
// Binary arrays are little-endian float32; JSONL contains byte offsets/shapes.
#pragma once
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <vector>
#include "json.hpp"

struct AuditTensor {
    std::string name;
    const float *data;
    size_t size;
    std::vector<size_t> shape;
};
template <class T>
AuditTensor audit_tensor(const std::string &name, const T &values, std::vector<size_t> shape) {
    return {name, values.data(), static_cast<size_t>(values.size()), std::move(shape)};
}

class ObservationAudit {
    std::ofstream data_, index_;
    std::filesystem::path directory_;
    size_t count_ = 0, maximum_ = 5000;
public:
    ObservationAudit() {
        const char *root = std::getenv("JEPLO_OBSERVATION_AUDIT_DIR");
        if (!root || !*root) throw std::runtime_error("Diagnostic controller requires JEPLO_OBSERVATION_AUDIT_DIR");
        uint16_t endian = 1;
        if (*reinterpret_cast<uint8_t*>(&endian) != 1) throw std::runtime_error("Audit requires little-endian host");
        std::filesystem::path directory(root);
        directory_ = directory;
        if (std::filesystem::exists(directory)) throw std::runtime_error("Audit directory already exists");
        std::filesystem::create_directories(directory);
        if (const char *n = std::getenv("JEPLO_OBSERVATION_AUDIT_MAX_RECORDS")) {
            maximum_ = std::stoul(n);
            if (!maximum_) throw std::runtime_error("Audit record limit must be positive");
        }
        data_.open(directory / "tensors.bin", std::ios::binary);
        index_.open(directory / "records.jsonl");
        if (!data_ || !index_) throw std::runtime_error("Cannot open audit files");
        nlohmann::json metadata = {{"format","jeplo-observation-trace-v1"}, {"side","mujoco"},
            {"clock","wall_clock_ns"}, {"scan_time_semantics","publisher receive time, not original acquisition"},
            {"source_sha256",JEPLO_AUDIT_SOURCE_SHA}, {"max_records",maximum_}};
        std::ofstream(directory / "metadata.json") << metadata.dump(2) << '\n';
    }
    void configuration(const nlohmann::json &value) {
        std::ofstream(directory_ / "runtime.json") << value.dump(2) << '\n';
    }
    void record(const char *kind, int step, nlohmann::json metadata,
                std::initializer_list<AuditTensor> tensors) {
        if (count_ >= maximum_) return;
        metadata["kind"] = kind;
        metadata["step"] = step;
        metadata["arrays"] = nlohmann::json::object();
        for (const auto &tensor : tensors) {
            size_t count=1;
            for(auto dimension:tensor.shape) count*=dimension;
            if(count!=tensor.size) throw std::runtime_error("Audit tensor shape mismatch: "+tensor.name);
            metadata["arrays"][tensor.name] = {{"offset",static_cast<size_t>(data_.tellp())},{"shape",tensor.shape}};
            data_.write(reinterpret_cast<const char*>(tensor.data),tensor.size*sizeof(float));
        }
        data_.flush();
        if (!data_) throw std::runtime_error("Audit tensor write failed");
        index_ << metadata.dump() << '\n';
        index_.flush();
        if (!index_) throw std::runtime_error("Audit index write failed");
        ++count_;
    }
};
