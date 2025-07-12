#pragma once
#include <string>
#include <cstddef>
#include <chrono>
#include <memory>
#include <atomic>

namespace hshm::lbm {

enum class TransportType {
    AUTO = 0,
    TCP = 1,
    RDMA = 2
};

enum class OperationType {
    SEND = 0,
    RECV = 1,
    RMA_WRITE = 2,
    RMA_READ = 3
};

struct Event {
    bool is_done = false;
    int error_code = 0;
    std::string error_message;
    size_t bytes_transferred = 0;
    TransportType transport_used = TransportType::AUTO;
    OperationType operation_type = OperationType::SEND;
    
    // Timing support for timeouts
    std::chrono::steady_clock::time_point start_time;
    std::chrono::milliseconds timeout{5000}; // 5 second default timeout
    
    // Event ID for tracking
    uint64_t event_id = 0;
    
    // Context pointer for user data
    void* user_context = nullptr;
    
    bool has_timed_out() const {
        auto now = std::chrono::steady_clock::now();
        return (now - start_time) > timeout;
    }
};

struct MemoryRegion {
    void* addr = nullptr;
    size_t length = 0;
    uint64_t key = 0;  // Remote key for RDMA
    void* desc = nullptr;  // Local descriptor
    void* mr = nullptr;  // libfabric memory region - use void* to avoid forward declaration issues
    
    // For cleanup
    ~MemoryRegion();
    
    // Non-copyable but movable
    MemoryRegion() = default;
    MemoryRegion(const MemoryRegion&) = delete;
    MemoryRegion& operator=(const MemoryRegion&) = delete;
    MemoryRegion(MemoryRegion&& other) noexcept;
    MemoryRegion& operator=(MemoryRegion&& other) noexcept;
};

struct Bulk {
    char* data = nullptr;
    size_t size = 0;
    std::string target_url;
    TransportType preferred_transport = TransportType::AUTO;
    
    // For ZMQ
    void* zmq_handle = nullptr;
    
    // For RDMA - memory registration info
    std::shared_ptr<MemoryRegion> local_mr;
    std::shared_ptr<MemoryRegion> remote_mr;  // For RMA operations
    
    // Message header for proper size handling
    struct MessageHeader {
        uint32_t magic = 0xDEADBEEF;
        uint32_t size = 0;
        uint64_t sequence = 0;
    } __attribute__((packed));
    
    // Get total message size including header
    size_t total_size() const {
        return sizeof(MessageHeader) + size;
    }
};

} // namespace hshm::lbm