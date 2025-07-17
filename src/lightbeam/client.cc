#include "hermes_shm/lightbeam/client.h"
#include "hermes_shm/lightbeam/libfabric/client.h"
#include "hermes_shm/lightbeam/zmq/client.h"
#include <memory>
#include <iostream>
#include <vector>

namespace hshm::lbm {

enum class ClientType {
    UNKNOWN,
    ZMQ,
    LIBFABRIC
};

struct Client::Impl {
    std::unique_ptr<libfabric::Client> libfabric_client;
    std::unique_ptr<zmq::Client> zmq_client;
    ClientType current_type = ClientType::UNKNOWN;
    
    // Store events from ZMQ client (which returns unique_ptr<Event>)
    std::vector<std::unique_ptr<Event>> zmq_events;
    
    ClientType determine_client_type(const std::string& url) {
        if (url.find(":5555") != std::string::npos) {
            return ClientType::ZMQ;  // ZMQ port
        } else if (url.find(":5556") != std::string::npos) {
            return ClientType::LIBFABRIC;  // LibFabric port
        } else {
            // Default to ZMQ for unknown ports
            return ClientType::ZMQ;
        }
    }
};

Client::Client() : impl_(std::make_unique<Impl>()) {}

Client::~Client() {
    // Destructor must be defined in .cc file for PIMPL to work properly
}

void Client::Connect(const std::string &url) {
    if (!impl_) {
        impl_ = std::make_unique<Impl>();
    }
    
    impl_->current_type = impl_->determine_client_type(url);
    
    if (impl_->current_type == ClientType::ZMQ) {
        std::cout << "[Main Client] Using ZMQ implementation for " << url << std::endl;
        if (!impl_->zmq_client) {
            impl_->zmq_client = std::make_unique<zmq::Client>();
        }
        impl_->zmq_client->Connect(url, TransportType::TCP);
    } else {
        std::cout << "[Main Client] Using LibFabric implementation for " << url << std::endl;
        if (!impl_->libfabric_client) {
            impl_->libfabric_client = std::make_unique<libfabric::Client>();
        }
        impl_->libfabric_client->Connect(url);
    }
}

void Client::Disconnect(const std::string &url) {
    if (impl_->current_type == ClientType::ZMQ && impl_->zmq_client) {
        impl_->zmq_client->Disconnect(url);
    } else if (impl_->current_type == ClientType::LIBFABRIC && impl_->libfabric_client) {
        impl_->libfabric_client->Disconnect(url);
    }
}

Bulk Client::Expose(const std::string &url, const char *data, size_t data_size, int flags) {
    if (!impl_) {
        impl_ = std::make_unique<Impl>();
    }
    
    if (impl_->current_type == ClientType::ZMQ && impl_->zmq_client) {
        return impl_->zmq_client->Expose(url, data, data_size, flags);
    } else if (impl_->current_type == ClientType::LIBFABRIC && impl_->libfabric_client) {
        return impl_->libfabric_client->Expose(url, data, data_size, flags);
    }
    
    // Fallback
    Bulk bulk;
    bulk.data = const_cast<char*>(data);
    bulk.size = data_size;
    return bulk;
}

Event* Client::Send(const Bulk &bulk) {
    if (!impl_) {
        impl_ = std::make_unique<Impl>();
    }
    
    if (impl_->current_type == ClientType::ZMQ && impl_->zmq_client) {
        auto event = impl_->zmq_client->Send(bulk);
        Event* raw_ptr = event.get();
        impl_->zmq_events.push_back(std::move(event)); // Store ownership
        return raw_ptr;
    } else if (impl_->current_type == ClientType::LIBFABRIC && impl_->libfabric_client) {
        return impl_->libfabric_client->Send(bulk);
    }
    
    return nullptr;
}

Event* Client::Recv(char *buffer, size_t buffer_size, const std::string &from_url) {
    if (!impl_) {
        impl_ = std::make_unique<Impl>();
    }
    
    if (impl_->current_type == ClientType::ZMQ && impl_->zmq_client) {
        auto event = impl_->zmq_client->Recv(buffer, buffer_size, from_url);
        Event* raw_ptr = event.get();
        impl_->zmq_events.push_back(std::move(event)); // Store ownership
        return raw_ptr;
    } else if (impl_->current_type == ClientType::LIBFABRIC && impl_->libfabric_client) {
        return impl_->libfabric_client->Recv(buffer, buffer_size, from_url);
    }
    
    return nullptr;
}

void Client::ProcessCompletions() {
    if (impl_->current_type == ClientType::ZMQ && impl_->zmq_client) {
        impl_->zmq_client->ProcessCompletions();
    } else if (impl_->current_type == ClientType::LIBFABRIC && impl_->libfabric_client) {
        impl_->libfabric_client->ProcessCompletions();
    }
}

} // namespace hshm::lbm 