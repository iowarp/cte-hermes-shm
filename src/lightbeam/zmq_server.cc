#include "hermes_shm/lightbeam/zmq/server.h"
#include "hermes_shm/lightbeam/types.h"
#include <zmq.h>
#include <iostream>
#include <cstring>

namespace hshm::lbm::zmq {

struct Server::Impl {
    void* context = nullptr;
    void* socket = nullptr;
    bool running = false;
    char recv_buffer[1024];
    char send_buffer[1024];
    char identity_buffer[256]; // For ROUTER socket identity
};

Server::Server() : impl_(std::make_unique<Impl>()) {}
Server::~Server() = default;

void Server::StartServer(const std::string& url, hshm::lbm::TransportType transport) {
    std::cout << "[ZMQ Server] Starting server on " << url << std::endl;
    
    impl_->context = zmq_ctx_new();
    
    // Use ROUTER socket to match DEALER clients
    impl_->socket = zmq_socket(impl_->context, ZMQ_ROUTER);
    
    int rc = zmq_bind(impl_->socket, url.c_str());
    if (rc != 0) {
        std::cerr << "[ZMQ Server] Bind failed: " << zmq_strerror(zmq_errno()) << std::endl;
        return;
    }
    
    impl_->running = true;
    std::cout << "[ZMQ Server] Server started successfully" << std::endl;
}

void Server::Stop() {
    std::cout << "[ZMQ Server] Stopping server" << std::endl;
    impl_->running = false;
    
    if (impl_->socket) {
        zmq_close(impl_->socket);
        impl_->socket = nullptr;
    }
    if (impl_->context) {
        zmq_ctx_term(impl_->context);
        impl_->context = nullptr;
    }
}

void Server::ProcessMessages() {
    if (!impl_->running || !impl_->socket) return;
    
    zmq_pollitem_t items[] = {{impl_->socket, 0, ZMQ_POLLIN, 0}};
    int rc = zmq_poll(items, 1, 0); // Non-blocking poll
    
    if (rc > 0 && (items[0].revents & ZMQ_POLLIN)) {
        // ROUTER socket: first frame is client identity, second is message
        
        // Receive client identity
        int identity_size = zmq_recv(impl_->socket, impl_->identity_buffer, sizeof(impl_->identity_buffer) - 1, 0);
        if (identity_size <= 0) {
            std::cerr << "[ZMQ Server] Failed to receive client identity" << std::endl;
            return;
        }
        
        // Check if there's more data
        int more;
        size_t more_size = sizeof(more);
        zmq_getsockopt(impl_->socket, ZMQ_RCVMORE, &more, &more_size);
        
        if (!more) {
            std::cerr << "[ZMQ Server] No message data received" << std::endl;
            return;
        }
        
        // Receive message
        int n = zmq_recv(impl_->socket, impl_->recv_buffer, sizeof(impl_->recv_buffer) - 1, 0);
        if (n > 0) {
            std::cout << "[ZMQ Server] Received " << n << " bytes from client" << std::endl;
            
            // Parse message header if present
            const char* payload = impl_->recv_buffer;
            size_t payload_size = n;
            
            if (n >= static_cast<int>(sizeof(hshm::lbm::Bulk::MessageHeader))) {
                auto* header = reinterpret_cast<hshm::lbm::Bulk::MessageHeader*>(impl_->recv_buffer);
                if (header->magic == 0xDEADBEEF) {
                    // Valid message with header
                    payload = impl_->recv_buffer + sizeof(hshm::lbm::Bulk::MessageHeader);
                    payload_size = std::min(static_cast<size_t>(header->size), 
                                          static_cast<size_t>(n) - sizeof(hshm::lbm::Bulk::MessageHeader));
                    
                    std::cout << "[ZMQ Server] Parsed message header: magic=0x" << std::hex << header->magic 
                              << ", size=" << std::dec << header->size 
                              << ", sequence=" << header->sequence << std::endl;
                }
            }
            
            // Create response with header
            auto* response_header = reinterpret_cast<hshm::lbm::Bulk::MessageHeader*>(impl_->send_buffer);
            response_header->magic = 0xDEADBEEF;
            response_header->size = payload_size;
            response_header->sequence = 0; // Server response
            
            // Copy payload to response
            std::memcpy(impl_->send_buffer + sizeof(hshm::lbm::Bulk::MessageHeader), 
                       payload, payload_size);
            
            size_t total_response_size = sizeof(hshm::lbm::Bulk::MessageHeader) + payload_size;
            
            // Send response back to client (ROUTER: first identity, then message)
            zmq_send(impl_->socket, impl_->identity_buffer, identity_size, ZMQ_SNDMORE);
            zmq_send(impl_->socket, impl_->send_buffer, total_response_size, 0);
            
            // Create null-terminated string for logging
            char log_buffer[256];
            size_t log_size = std::min(payload_size, sizeof(log_buffer) - 1);
            std::memcpy(log_buffer, payload, log_size);
            log_buffer[log_size] = '\0';
            
            std::cout << "[ZMQ Server] Echoed back payload: \"" << log_buffer << "\" (" 
                      << payload_size << " bytes, total with header: " << total_response_size << " bytes)" << std::endl;
        }
    }
}

// NEW: Recv method implementation
bool Server::Recv(const Bulk &bulk) {
    if (!impl_->running || !impl_->socket) return false;
    
    // Non-blocking check for incoming messages
    zmq_pollitem_t items[] = {{impl_->socket, 0, ZMQ_POLLIN, 0}};
    int rc = zmq_poll(items, 1, 0); // Non-blocking poll
    
    if (rc > 0 && (items[0].revents & ZMQ_POLLIN)) {
        char identity_buffer[256];
        
        // Receive client identity
        int identity_size = zmq_recv(impl_->socket, identity_buffer, sizeof(identity_buffer) - 1, 0);
        if (identity_size <= 0) {
            return false;
        }
        
        // Check if there's more data
        int more;
        size_t more_size = sizeof(more);
        zmq_getsockopt(impl_->socket, ZMQ_RCVMORE, &more, &more_size);
        
        if (!more) {
            return false;
        }
        
        // Receive message into bulk buffer
        int n = zmq_recv(impl_->socket, bulk.data, bulk.size - 1, 0);
        if (n > 0) {
            bulk.data[n] = '\0'; // Null terminate
            
            std::cout << "[ZMQ Server] Recv function received " << n << " bytes" << std::endl;
            
            // Parse message if it has header
            if (n >= static_cast<int>(sizeof(hshm::lbm::Bulk::MessageHeader))) {
                auto* header = reinterpret_cast<hshm::lbm::Bulk::MessageHeader*>(bulk.data);
                if (header->magic == 0xDEADBEEF) {
                    // Move payload to start of buffer
                    size_t payload_size = std::min(static_cast<size_t>(header->size), 
                                                  static_cast<size_t>(n) - sizeof(hshm::lbm::Bulk::MessageHeader));
                    std::memmove(bulk.data, bulk.data + sizeof(hshm::lbm::Bulk::MessageHeader), payload_size);
                    bulk.data[payload_size] = '\0';
                    
                    std::cout << "[ZMQ Server] Extracted payload: " << payload_size << " bytes" << std::endl;
                }
            }
            
            return true; // Successfully received a message
        }
    }
    
    return false; // No message received
}

bool Server::IsRunning() const {
    return impl_->running;
}

} // namespace hshm::lbm::zmq