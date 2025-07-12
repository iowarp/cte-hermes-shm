#include "hermes_shm/lightbeam/client.h"
#include "hermes_shm/lightbeam/transport_factory.h"
#include "hermes_shm/lightbeam/base_client.h"

namespace hshm::lbm {

struct Client::Impl {
    std::unique_ptr<IClient> client_;
    TransportType current_transport_ = TransportType::AUTO;
};

Client::Client(TransportType transport) : impl_(std::make_unique<Impl>()) {
    impl_->current_transport_ = transport;
    impl_->client_ = Transport::CreateClient(transport);
}

Client::~Client() = default;

void Client::Connect(const std::string &url, TransportType transport) {
    // Update transport if different from constructor
    if (transport != impl_->current_transport_) {
        impl_->current_transport_ = transport;
        impl_->client_ = Transport::CreateClient(transport);
    }
    
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

std::unique_ptr<Event> Client::Send(const Bulk &bulk) {
    if (impl_->client_) {
        return impl_->client_->Send(bulk);
    }
    return nullptr;
}

std::unique_ptr<Event> Client::Recv(char *buffer, size_t buffer_size, const std::string &from_url) {
    if (impl_->client_) {
        return impl_->client_->Recv(buffer, buffer_size, from_url);
    }
    return nullptr;
}

void Client::ProcessCompletions(double timeout_msec) {
    if (impl_->client_) {
        impl_->client_->ProcessCompletions(timeout_msec);
    }
}

} // namespace hshm::lbm 