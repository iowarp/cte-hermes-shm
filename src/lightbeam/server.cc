// Updated server.cc to route between ZMQ and LibFabric implementations
#include "hermes_shm/lightbeam/server.h"
#include "hermes_shm/lightbeam/libfabric/server.h"
#include "hermes_shm/lightbeam/zmq/server.h"
#include <memory>
#include <iostream>
#include <thread>
#include <chrono>

namespace hshm::lbm {

enum class ServerType {
    UNKNOWN,
    ZMQ,
    LIBFABRIC
};

struct Server::Impl {
    std::unique_ptr<libfabric::Server> libfabric_server;
    std::unique_ptr<zmq::Server> zmq_server;
    ServerType current_type = ServerType::UNKNOWN;
    
    ServerType determine_server_type(const std::string& url) {
        if (url.find(":5555") != std::string::npos) {
            return ServerType::ZMQ;  // ZMQ port
        } else if (url.find(":5556") != std::string::npos) {
            return ServerType::LIBFABRIC;  // LibFabric port
        } else {
            // Default to ZMQ for unknown ports
            return ServerType::ZMQ;
        }
    }
};

Server::Server() : impl_(std::make_unique<Impl>()) {}

Server::~Server() = default;

void Server::StartServer(const std::string &url) {
    impl_->current_type = impl_->determine_server_type(url);
    
    if (impl_->current_type == ServerType::ZMQ) {
        std::cout << "[Main Server] Using ZMQ implementation for " << url << std::endl;
        impl_->zmq_server = std::make_unique<zmq::Server>();
        impl_->zmq_server->StartServer(url, TransportType::TCP);
        
        // Start message processing loop for ZMQ server
        std::thread zmq_loop([this]() {
            while (impl_->zmq_server && impl_->zmq_server->IsRunning()) {
                impl_->zmq_server->ProcessMessages();
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        });
        zmq_loop.detach();
        
    } else {
        std::cout << "[Main Server] Using LibFabric implementation for " << url << std::endl;
        impl_->libfabric_server = std::make_unique<libfabric::Server>();
        impl_->libfabric_server->StartServer(url);
    }
}

void Server::Stop() {
    if (impl_->current_type == ServerType::ZMQ && impl_->zmq_server) {
        impl_->zmq_server->Stop();
    } else if (impl_->current_type == ServerType::LIBFABRIC && impl_->libfabric_server) {
        impl_->libfabric_server->Stop();
    }
}

bool Server::IsRunning() const {
    if (impl_->current_type == ServerType::ZMQ && impl_->zmq_server) {
        return impl_->zmq_server->IsRunning();
    } else if (impl_->current_type == ServerType::LIBFABRIC && impl_->libfabric_server) {
        return impl_->libfabric_server->IsRunning();
    }
    return false;
}

} // namespace hshm::lbm