#include "hermes_shm/lightbeam/client.h"
#include "hermes_shm/lightbeam/transport_factory.h"
#include "hermes_shm/lightbeam/base_client.h"

namespace hshm::lbm {

struct Client::Impl {
    std::unique_ptr<IClient> client_;
    TransportType current_transport_ = TransportType::AUTO;
};

Client::Client() : impl_(std::make_unique<Impl>()) {}

Client::~Client() = default;

void Client::Connect(const std::string &url, TransportType transport) {
    // Auto-detect transport type if needed
    if (transport == TransportType::AUTO) {
        // Default to TCP for now, could be made smarter
        transport = TransportType::TCP;
    }
    
    // Create the appropriate transport implementation
    impl_->client_ = Transport::CreateClient(transport);
    impl_->current_transport_ = transport;
    
    if (impl_->client_) {
        impl_->client_->Connect(url, transport);
    }
}

void Client::Disconnect(const std::string &url) {
    if (impl_->client_) {
        impl_->client_->Disconnect(url);
    }
}

Bulk Client::Expose(const std::string &url, const char *data, size_t data_size, int flags) {
    if (impl_->client_) {
        return impl_->client_->Expose(url, data, data_size, flags);
    }
    return {}; // Return empty Bulk on error
}

Event* Client::Send(const Bulk &bulk) {
    if (impl_->client_) {
        return impl_->client_->Send(bulk);
    }
    return nullptr;
}

Event* Client::Recv(char *buffer, size_t buffer_size, const std::string &from_url) {
    if (impl_->client_) {
        return impl_->client_->Recv(buffer, buffer_size, from_url);
    }
    return nullptr;
}

void Client::ProcessCompletions() {
    if (impl_->client_) {
        impl_->client_->ProcessCompletions();
    }
}

} // namespace hshm::lbm 