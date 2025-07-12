#include "hermes_shm/lightbeam/types.h"
#include <rdma/fi_domain.h>
#include <iostream>

namespace hshm::lbm {

// MemoryRegion destructor implementation
MemoryRegion::~MemoryRegion() {
    if (mr) {
        auto* fid_mr_ptr = static_cast<struct fid_mr*>(mr);
        fi_close(&fid_mr_ptr->fid);
        mr = nullptr;
    }
}

// Move constructor
MemoryRegion::MemoryRegion(MemoryRegion&& other) noexcept
    : addr(other.addr)
    , length(other.length)
    , key(other.key)
    , desc(other.desc)
    , mr(other.mr) {
    // Reset other object
    other.addr = nullptr;
    other.length = 0;
    other.key = 0;
    other.desc = nullptr;
    other.mr = nullptr;
}

// Move assignment operator
MemoryRegion& MemoryRegion::operator=(MemoryRegion&& other) noexcept {
    if (this != &other) {
        // Clean up current resources
        if (mr) {
            auto* fid_mr_ptr = static_cast<struct fid_mr*>(mr);
            fi_close(&fid_mr_ptr->fid);
        }
        
        // Transfer ownership
        addr = other.addr;
        length = other.length;
        key = other.key;
        desc = other.desc;
        mr = other.mr;
        
        // Reset other object
        other.addr = nullptr;
        other.length = 0;
        other.key = 0;
        other.desc = nullptr;
        other.mr = nullptr;
    }
    return *this;
}

} // namespace hshm::lbm