#include "hermes_shm/lightbeam/zmq/client.h"
#include <zmq.h>
#include <iostream>
#include <cstring>
#include <queue>
#include <unordered_map>
#include <memory>
#include <atomic>
#include <chrono>

namespace hshm::lbm::zmq {

struct PendingOperation {
    Event* event; // Raw pointer to the same event returned to user
    OperationType op_type;
    std::unique_ptr<char[]> buffer; // For send operations
    size_t buffer_size;
    char* user_buffer = nullptr; // For receive operations
    size_t user_buffer_size = 0;
};

struct Client::Impl {
    void* context = nullptr;
    void* socket = nullptr;
    bool connected = false;
    
    // Event management - proper queue for multiple concurrent operations
    std::atomic<uint64_t> next_event_id{1};
    std::queue<std::unique_ptr<PendingOperation>> pending_operations;
    std::unordered_map<uint64_t, Event*> active_events;
    
    uint64_t get_next_event_id() {
        return next_event_id.fetch_add(1);
    }
};

Client::Client() : impl_(std::make_unique<Impl>()) {}
Client::~Client() = default;

void Client::Connect(const std::string& url, hshm::lbm::TransportType transport) {
    std::cout << "[ZMQ Client] Connecting to " << url << std::endl;
    impl_->context = zmq_ctx_new();
    
    // Use DEALER socket for multiple concurrent messages
    impl_->socket = zmq_socket(impl_->context, ZMQ_DEALER);
    
    // Set socket to non-blocking mode
    int timeout = 0;
    zmq_setsockopt(impl_->socket, ZMQ_RCVTIMEO, &timeout, sizeof(timeout));
    zmq_setsockopt(impl_->socket, ZMQ_SNDTIMEO, &timeout, sizeof(timeout));
    
    int rc = zmq_connect(impl_->socket, url.c_str());
    if (rc != 0) {
        std::cerr << "[ZMQ Client] Connect failed: " << zmq_strerror(zmq_errno()) << std::endl;
        return;
    }
    impl_->connected = true;
    std::cout << "[ZMQ Client] Connected successfully" << std::endl;
}

void Client::Disconnect(const std::string& url) {
    std::cout << "[ZMQ Client] Disconnecting from " << url << std::endl;
    impl_->connected = false;
    
    // Clear all pending operations and active events
    while (!impl_->pending_operations.empty()) {
        impl_->pending_operations.pop();
    }
    impl_->active_events.clear();
    
    if (impl_->socket) {
        zmq_close(impl_->socket);
        impl_->socket = nullptr;
    }
    if (impl_->context) {
        zmq_ctx_term(impl_->context);
        impl_->context = nullptr;
    }
}

hshm::lbm::Bulk Client::Expose(const std::string& url, const char* data, size_t data_size, int flags) {
    std::cout << "[ZMQ Client] Exposing data of size " << data_size << std::endl;
    
    hshm::lbm::Bulk bulk;
    bulk.data = const_cast<char*>(data);
    bulk.size = data_size;
    bulk.target_url = url;
    bulk.preferred_transport = hshm::lbm::TransportType::TCP;
    bulk.zmq_handle = impl_->socket;
    return bulk;
}

std::unique_ptr<hshm::lbm::Event> Client::Send(const hshm::lbm::Bulk& bulk) {
    // Create new heap-allocated event
    auto event = std::make_unique<Event>();
    event->event_id = impl_->get_next_event_id();
    event->operation_type = OperationType::SEND;
    event->transport_used = hshm::lbm::TransportType::TCP;
    event->start_time = std::chrono::steady_clock::now();
    event->is_done = false;
    event->error_code = 0;
    
    if (!impl_->connected || !impl_->socket) {
        event->is_done = true;
        event->error_code = -1;
        event->error_message = "Not connected";
        return event;
    }
    
    std::cout << "[ZMQ Client] Queuing async send of " << bulk.size << " bytes (event_id=" << event->event_id << ")" << std::endl;
    
    // Create message with header
    auto total_size = bulk.total_size();
    auto message_buffer = std::make_unique<char[]>(total_size);
    
    // Fill header
    auto* header = reinterpret_cast<Bulk::MessageHeader*>(message_buffer.get());
    header->magic = 0xDEADBEEF;
    header->size = bulk.size;
    header->sequence = event->event_id;
    
    // Copy payload
    std::memcpy(message_buffer.get() + sizeof(Bulk::MessageHeader), bulk.data, bulk.size);
    
    // Create pending operation and queue it
    auto pending_op = std::make_unique<PendingOperation>();
    pending_op->event = event.get(); // Share the same event
    pending_op->op_type = OperationType::SEND;
    pending_op->buffer = std::move(message_buffer);
    pending_op->buffer_size = total_size;
    
    impl_->pending_operations.push(std::move(pending_op));
    
    // Store event for user tracking
    uint64_t event_id = event->event_id;
    impl_->active_events[event_id] = event.get();
    
    // Return the shared event as unique_ptr (transfer ownership)
    return std::unique_ptr<Event>(event.release());
}

std::unique_ptr<hshm::lbm::Event> Client::Recv(char* buffer, size_t buffer_size, const std::string& from_url) {
    // Create new heap-allocated event
    auto event = std::make_unique<Event>();
    event->event_id = impl_->get_next_event_id();
    event->operation_type = OperationType::RECV;
    event->transport_used = hshm::lbm::TransportType::TCP;
    event->start_time = std::chrono::steady_clock::now();
    event->is_done = false;
    event->error_code = 0;
    
    if (!impl_->connected || !impl_->socket) {
        event->is_done = true;
        event->error_code = -1;
        event->error_message = "Not connected";
        return std::unique_ptr<Event>(event.release());
    }
    
    if (!buffer || buffer_size == 0) {
        event->is_done = true;
        event->error_code = -1;
        event->error_message = "Invalid buffer";
        return std::unique_ptr<Event>(event.release());
    }
    
    std::cout << "[ZMQ Client] Queuing async receive (event_id=" << event->event_id << ")" << std::endl;
    
    // Create pending operation and queue it
    auto pending_op = std::make_unique<PendingOperation>();
    pending_op->event = event.get(); // Share the same event
    pending_op->op_type = OperationType::RECV;
    pending_op->buffer = nullptr;
    pending_op->buffer_size = 0;
    pending_op->user_buffer = buffer;
    pending_op->user_buffer_size = buffer_size;
    
    impl_->pending_operations.push(std::move(pending_op));
    
    // Store event for user tracking
    uint64_t event_id = event->event_id;
    impl_->active_events[event_id] = event.get();
    
    // Return the shared event as unique_ptr (transfer ownership)
    return std::unique_ptr<Event>(event.release());
}

void Client::ProcessCompletions(double timeout_msec) {
    if (!impl_->connected || !impl_->socket) return;
    
    std::queue<std::unique_ptr<PendingOperation>> remaining_operations;
    bool any_completed = false;
    
    // Process all pending operations
    while (!impl_->pending_operations.empty()) {
        auto& pending_op = impl_->pending_operations.front();
        bool operation_completed = false;
        
        if (pending_op->op_type == OperationType::SEND) {
            // Try to send the message
            int rc = zmq_send(impl_->socket, pending_op->buffer.get(), pending_op->buffer_size, ZMQ_DONTWAIT);
            
            if (rc == static_cast<int>(pending_op->buffer_size)) {
                // Send successful
                pending_op->event->is_done = true;
                pending_op->event->error_code = 0;
                pending_op->event->bytes_transferred = rc;
                operation_completed = true;
                any_completed = true;
                std::cout << "[ZMQ Client] Send completed: " << rc << " bytes (event_id=" << pending_op->event->event_id << ")" << std::endl;
            } else if (rc == -1) {
                int err = zmq_errno();
                if (err == EAGAIN) {
                    // Would block, check for timeout
                    if (pending_op->event->has_timed_out()) {
                        pending_op->event->is_done = true;
                        pending_op->event->error_code = -ETIMEDOUT;
                        pending_op->event->error_message = "Send timeout";
                        operation_completed = true;
                        any_completed = true;
                        std::cout << "[ZMQ Client] Send timed out (event_id=" << pending_op->event->event_id << ")" << std::endl;
                    }
                } else {
                    // Real error
                    pending_op->event->is_done = true;
                    pending_op->event->error_code = err;
                    pending_op->event->error_message = zmq_strerror(err);
                    operation_completed = true;
                    any_completed = true;
                    std::cout << "[ZMQ Client] Send failed: " << zmq_strerror(err) << " (event_id=" << pending_op->event->event_id << ")" << std::endl;
                }
            }
        } else if (pending_op->op_type == OperationType::RECV) {
            // Try to receive a message
            int rc = zmq_recv(impl_->socket, pending_op->user_buffer, pending_op->user_buffer_size - 1, ZMQ_DONTWAIT);
            
            if (rc > 0) {
                // Receive successful
                pending_op->user_buffer[rc] = '\0'; // Null terminate for safety
                
                // Parse message header if present
                if (rc >= static_cast<int>(sizeof(Bulk::MessageHeader))) {
                    auto* header = reinterpret_cast<Bulk::MessageHeader*>(pending_op->user_buffer);
                    if (header->magic == 0xDEADBEEF) {
                        // Valid message with header
                        size_t payload_size = std::min(static_cast<size_t>(header->size), 
                                                      static_cast<size_t>(rc) - sizeof(Bulk::MessageHeader));
                        
                        // Move payload to start of buffer
                        std::memmove(pending_op->user_buffer, pending_op->user_buffer + sizeof(Bulk::MessageHeader), payload_size);
                        pending_op->user_buffer[payload_size] = '\0';
                        
                        pending_op->event->bytes_transferred = payload_size;
                        std::cout << "[ZMQ Client] Received valid message: " << payload_size << " bytes (event_id=" << pending_op->event->event_id << ")" << std::endl;
                    } else {
                        // Raw data without header
                        pending_op->event->bytes_transferred = rc;
                        std::cout << "[ZMQ Client] Received raw data: " << rc << " bytes (event_id=" << pending_op->event->event_id << ")" << std::endl;
                    }
                } else {
                    // Small message, assume raw data
                    pending_op->event->bytes_transferred = rc;
                    std::cout << "[ZMQ Client] Received small message: " << rc << " bytes (event_id=" << pending_op->event->event_id << ")" << std::endl;
                }
                
                pending_op->event->is_done = true;
                pending_op->event->error_code = 0;
                operation_completed = true;
                any_completed = true;
            } else if (rc == -1) {
                int err = zmq_errno();
                if (err == EAGAIN) {
                    // Would block, check for timeout
                    if (pending_op->event->has_timed_out()) {
                        pending_op->event->is_done = true;
                        pending_op->event->error_code = -ETIMEDOUT;
                        pending_op->event->error_message = "Receive timeout";
                        operation_completed = true;
                        any_completed = true;
                        std::cout << "[ZMQ Client] Receive timed out (event_id=" << pending_op->event->event_id << ")" << std::endl;
                    }
                } else {
                    // Real error
                    pending_op->event->is_done = true;
                    pending_op->event->error_code = err;
                    pending_op->event->error_message = zmq_strerror(err);
                    operation_completed = true;
                    any_completed = true;
                    std::cout << "[ZMQ Client] Receive failed: " << zmq_strerror(err) << " (event_id=" << pending_op->event->event_id << ")" << std::endl;
                }
            }
        }
        
        if (!operation_completed) {
            // Keep operation for next iteration
            remaining_operations.push(std::move(pending_op));
        }
        
        impl_->pending_operations.pop();
    }
    
    // Put remaining operations back
    impl_->pending_operations = std::move(remaining_operations);
    
    // Clean up completed events from active_events map
    auto it = impl_->active_events.begin();
    while (it != impl_->active_events.end()) {
        if (it->second->is_done) {
            it = impl_->active_events.erase(it);
        } else {
            ++it;
        }
    }
    
    if (any_completed) {
        std::cout << "[ZMQ Client] ProcessCompletions completed, " 
                  << impl_->pending_operations.size() << " operations still pending" << std::endl;
    }
}

} // namespace hshm::lbm::zmq