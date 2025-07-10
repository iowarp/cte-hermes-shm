#include "hermes_shm/lightbeam/zmq/client.h"
#include <zmq.h>
#include <iostream>
#include <cstring>

namespace hshm::lbm::zmq {

struct Client::Impl {
    void* context = nullptr;
    void* socket = nullptr;
    bool connected = false;
    char recv_buffer[1024];
};

static hshm::lbm::Event dummy_event;

Client::Client() : impl_(std::make_unique<Impl>()) {}
Client::~Client() = default;

void Client::Connect(const std::string& url, hshm::lbm::TransportType transport) {
    std::cout << "[ZMQ Client] Connecting to " << url << std::endl;
    
    impl_->context = zmq_ctx_new();
    impl_->socket = zmq_socket(impl_->context, ZMQ_REQ);
    
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

hshm::lbm::Event* Client::Send(const hshm::lbm::Bulk& bulk) {
    if (!impl_->connected || !impl_->socket) {
        dummy_event.is_done = false;
        dummy_event.error_code = -1;
        dummy_event.error_message = "Not connected";
        dummy_event.bytes_transferred = 0;
        dummy_event.transport_used = hshm::lbm::TransportType::TCP;
        return &dummy_event;
    }
    
    std::cout << "[ZMQ Client] Sending: " << bulk.data << std::endl;
    
    int rc = zmq_send(impl_->socket, bulk.data, bulk.size, 0);
    
    dummy_event.is_done = (rc == (int)bulk.size);
    dummy_event.error_code = (rc < 0) ? zmq_errno() : 0;
    dummy_event.error_message = (rc < 0) ? zmq_strerror(zmq_errno()) : "";
    dummy_event.bytes_transferred = (rc > 0) ? rc : 0;
    dummy_event.transport_used = hshm::lbm::TransportType::TCP;
    
    return &dummy_event;
}

hshm::lbm::Event* Client::Recv(char* buffer, size_t buffer_size, const std::string& from_url) {
    if (!impl_->connected || !impl_->socket) {
        dummy_event.is_done = false;
        dummy_event.error_code = -1;
        dummy_event.error_message = "Not connected";
        dummy_event.bytes_transferred = 0;
        dummy_event.transport_used = hshm::lbm::TransportType::TCP;
        return &dummy_event;
    }
    
    std::cout << "[ZMQ Client] Receiving..." << std::endl;
    
    int rc = zmq_recv(impl_->socket, buffer, buffer_size - 1, 0);
    
    if (rc > 0) {
        buffer[rc] = '\0';
        std::cout << "[ZMQ Client] Received: " << buffer << std::endl;
    }
    
    dummy_event.is_done = (rc > 0);
    dummy_event.error_code = (rc < 0) ? zmq_errno() : 0;
    dummy_event.error_message = (rc < 0) ? zmq_strerror(zmq_errno()) : "";
    dummy_event.bytes_transferred = (rc > 0) ? rc : 0;
    dummy_event.transport_used = hshm::lbm::TransportType::TCP;
    
    return &dummy_event;
}

void Client::ProcessCompletions() {
    // For ZeroMQ, completions are handled synchronously in Send/Recv
    std::cout << "[ZMQ Client] ProcessCompletions called" << std::endl;
}

} // namespace hshm::lbm::zmq 