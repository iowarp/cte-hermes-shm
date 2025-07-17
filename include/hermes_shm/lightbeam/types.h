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
    RECV = 1
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

struct Bulk {
    char* data = nullptr;
    size_t size = 0;
    std::string target_url;
    
    // Flags to enable/disable RDMA
    int flags = 0;
    static const int RDMA_ENABLED = 1;
    
    // Internal transport details
    TransportType preferred_transport = TransportType::AUTO;
    void* transport_context = nullptr; // For libfabric-specific data
    
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