#include "hermes_shm/lightbeam/zmq/server.h"
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
};

Server::Server() : impl_(std::make_unique<Impl>()) {}
Server::~Server() = default;

void Server::StartServer(const std::string& url, hshm::lbm::TransportType transport) {
    std::cout << "[ZMQ Server] Starting server on " << url << std::endl;
    
    impl_->context = zmq_ctx_new();
    impl_->socket = zmq_socket(impl_->context, ZMQ_REP);
    
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
        // Receive message
        int n = zmq_recv(impl_->socket, impl_->recv_buffer, sizeof(impl_->recv_buffer) - 1, 0);
        if (n > 0) {
            impl_->recv_buffer[n] = '\0';
            std::cout << "[ZMQ Server] Received: " << impl_->recv_buffer << std::endl;
            
            // Echo back the message
            strcpy(impl_->send_buffer, impl_->recv_buffer);
            zmq_send(impl_->socket, impl_->send_buffer, strlen(impl_->send_buffer), 0);
            std::cout << "[ZMQ Server] Echoed back: " << impl_->send_buffer << std::endl;
        }
    }
}

bool Server::IsRunning() const {
    return impl_->running;
}

} // namespace hshm::lbm::zmq 