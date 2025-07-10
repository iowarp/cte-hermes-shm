#include "hermes_shm/lightbeam/server.h"
#include "hermes_shm/lightbeam/transport_factory.h"
#include "hermes_shm/lightbeam/base_server.h"

namespace hshm::lbm {

struct Server::Impl {
    std::unique_ptr<IServer> server_;
    TransportType current_transport_ = TransportType::AUTO;
};

Server::Server() : impl_(std::make_unique<Impl>()) {}

Server::~Server() = default;

void Server::StartServer(const std::string &url, TransportType transport) {
    // Auto-detect transport type if needed
    if (transport == TransportType::AUTO) {
        // Default to TCP for now, could be made smarter
        transport = TransportType::TCP;
    }
    
    // Create the appropriate transport implementation
    impl_->server_ = Transport::CreateServer(transport);
    impl_->current_transport_ = transport;
    
    if (impl_->server_) {
        impl_->server_->StartServer(url, transport);
    }
}

void Server::Stop() {
    if (impl_->server_) {
        impl_->server_->Stop();
    }
}

void Server::ProcessMessages() {
    if (impl_->server_) {
        impl_->server_->ProcessMessages();
    }
}

bool Server::IsRunning() const {
    if (impl_->server_) {
        return impl_->server_->IsRunning();
    }
    return false;
}

} // namespace hshm::lbm 