#pragma once

#include <cstdint>

namespace at {

struct PhiloxCudaState {
    PhiloxCudaState() = default;

    union Payload {
        uint64_t val;
        int64_t *ptr;
    };

    Payload seed_{};
    Payload offset_{};
    uint64_t offset_intragraph_ = 0;
    bool captured_ = false;
};

}  // namespace at
