#include "hermes_shm/lightbeam/libfabric/client.h"
#include "hermes_shm/lightbeam/libfabric/common.h"
#include "hermes_shm/lightbeam/utils.h"
#include <rdma/fabric.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_cm.h>
#include <iostream>
#include <cstring>
#include <thread>
#include <chrono>

namespace hshm::lbm::libfabric {

struct Client::Impl : public LibfabricCommon {
    struct fid_ep* endpoint = nullptr;
    bool connected = false;
};

static hshm::lbm::Event dummy_event;

Client::Client() : impl_(std::make_unique<Impl>()) {}
Client::~Client() {
    if (impl_) {
        impl_->connected = false;
        Disconnect("127.0.0.1:5556");
    }
}

void Client::Connect(const std::string& url, hshm::lbm::TransportType transport) {
    std::cout << "[Libfabric Client] Connecting to " << url << std::endl;
    
    auto [node, service] = utils::parseUrl(url);
    std::cout << "[Libfabric Client] Parsed address: " << node << ":" << service << std::endl;
    
    // Initialize common libfabric resources
    int rc = impl_->initializeResources(node, service, 0);
    if (rc != 0) {
        std::cerr << "[Libfabric Client] Failed to initialize libfabric resources" << std::endl;
        return;
    }
    
    // Create endpoint
    rc = fi_endpoint(impl_->domain_, impl_->info_, &impl_->endpoint, nullptr);
    if (rc != 0) {
        std::cerr << "[Libfabric Client] fi_endpoint failed: " << fi_strerror(rc) << std::endl;
        impl_->cleanup();
        return;
    }
    
    // Bind CQ and EQ to endpoint
    fi_ep_bind(impl_->endpoint, &impl_->cq_->fid, FI_TRANSMIT | FI_RECV);
    fi_ep_bind(impl_->endpoint, &impl_->eq_->fid, 0);
    
    // Enable endpoint
    fi_enable(impl_->endpoint);
    
    // Connect to server using the address from info
    std::cout << "[Libfabric Client] Initiating connection..." << std::endl;
    rc = fi_connect(impl_->endpoint, impl_->info_->dest_addr, nullptr, 0);
    if (rc != 0) {
        std::cerr << "[Libfabric Client] fi_connect failed: " << fi_strerror(rc) << std::endl;
        fi_close(&impl_->endpoint->fid);
        impl_->endpoint = nullptr;
        impl_->cleanup();
        return;
    }
    
    std::cout << "[Libfabric Client] fi_connect called successfully, waiting for events..." << std::endl;
    
    // Wait for connection completion using blocking read with timeout
    struct fi_eq_cm_entry entry;
    uint32_t event;
    bool connection_established = false;
    
    // Use fi_eq_sread for blocking read with timeout (in milliseconds)
    ssize_t eq_rc = fi_eq_sread(impl_->eq_, &event, &entry, sizeof(entry), 5000, 0); // 5 second timeout
    
    if (eq_rc > 0) {
        std::cout << "[Libfabric Client] Received event: " << event << std::endl;
        if (event == FI_CONNECTED) {
            impl_->connected = true;
            connection_established = true;
            std::cout << "[Libfabric Client] Connected successfully with provider: " << impl_->info_->fabric_attr->prov_name << std::endl;
        } else if (event == FI_SHUTDOWN) {
            std::cerr << "[Libfabric Client] Connection was shut down" << std::endl;
        } else {
            std::cout << "[Libfabric Client] Unexpected event: " << event << std::endl;
        }
    } else if (eq_rc == -FI_ETIMEDOUT) {
        std::cerr << "[Libfabric Client] Connection timeout" << std::endl;
    } else {
        std::cerr << "[Libfabric Client] Event queue error: " << fi_strerror(-eq_rc) << std::endl;
    }
    
    if (!connection_established) {
        std::cerr << "[Libfabric Client] Connection failed" << std::endl;
        // Clean up on failure
        if (impl_->endpoint) {
            fi_close(&impl_->endpoint->fid);
            impl_->endpoint = nullptr;
        }
        impl_->cleanup();
    }
}

void Client::Disconnect(const std::string& url) {
    if (!impl_->connected && !impl_->endpoint) return; // Already disconnected
    
    std::cout << "[Libfabric Client] Disconnecting from " << url << std::endl;
    impl_->connected = false;
    
    // Close endpoint first
    if (impl_->endpoint) {
        fi_close(&impl_->endpoint->fid);
        impl_->endpoint = nullptr;
    }
    
    // Clean up common resources
    impl_->cleanup();
}

hshm::lbm::Bulk Client::Expose(const std::string& url, const char* data, size_t data_size, int flags) {
    std::cout << "[Libfabric Client] Exposing data of size " << data_size << std::endl;
    
    hshm::lbm::Bulk bulk;
    bulk.data = const_cast<char*>(data);
    bulk.size = data_size;
    bulk.target_url = url;
    bulk.preferred_transport = hshm::lbm::TransportType::RDMA;
    bulk.zmq_handle = nullptr; // Not used for RDMA
    return bulk;
}

hshm::lbm::Event* Client::Send(const hshm::lbm::Bulk& bulk) {
    // Initialize dummy_event with safe defaults
    dummy_event.is_done = false;
    dummy_event.error_code = -1;
    dummy_event.error_message = "Not connected";
    dummy_event.bytes_transferred = 0;
    dummy_event.transport_used = hshm::lbm::TransportType::RDMA;
    
    if (!impl_) {
        dummy_event.error_message = "Client not initialized";
        return &dummy_event;
    }
    
    if (!impl_->connected || !impl_->endpoint) {
        dummy_event.error_message = "Not connected";
        return &dummy_event;
    }
    
    std::cout << "[Libfabric Client] Sending: " << bulk.data << std::endl;
    
    int rc = fi_send(impl_->endpoint, bulk.data, bulk.size, nullptr, FI_ADDR_UNSPEC, &impl_->send_ctx_);
    
    dummy_event.is_done = (rc == 0);
    dummy_event.error_code = (rc != 0) ? rc : 0;
    dummy_event.error_message = (rc != 0) ? fi_strerror(rc) : "";
    dummy_event.bytes_transferred = (rc == 0) ? bulk.size : 0;
    dummy_event.transport_used = hshm::lbm::TransportType::RDMA;
    
    return &dummy_event;
}

hshm::lbm::Event* Client::Recv(char* buffer, size_t buffer_size, const std::string& from_url) {
    // Initialize dummy_event with safe defaults
    dummy_event.is_done = false;
    dummy_event.error_code = -1;
    dummy_event.error_message = "Not connected";
    dummy_event.bytes_transferred = 0;
    dummy_event.transport_used = hshm::lbm::TransportType::RDMA;
    
    if (!impl_) {
        dummy_event.error_message = "Client not initialized";
        return &dummy_event;
    }
    
    if (!impl_->connected || !impl_->endpoint) {
        dummy_event.error_message = "Not connected";
        return &dummy_event;
    }
    
    if (!buffer || buffer_size == 0) {
        dummy_event.error_message = "Invalid buffer";
        return &dummy_event;
    }
    
    std::cout << "[Libfabric Client] Receiving..." << std::endl;
    
    // Post receive
    int rc = fi_recv(impl_->endpoint, buffer, buffer_size - 1, nullptr, FI_ADDR_UNSPEC, &impl_->recv_ctx_);
    if (rc != 0) {
        dummy_event.is_done = false;
        dummy_event.error_code = rc;
        dummy_event.error_message = fi_strerror(rc);
        dummy_event.bytes_transferred = 0;
        dummy_event.transport_used = hshm::lbm::TransportType::RDMA;
        return &dummy_event;
    }
    
    // Wait for completion with timeout and retry
    struct fi_cq_entry cq_entry;
    bool received = false;
    
    for (int attempt = 0; attempt < 50 && !received; ++attempt) { // Try for up to 5 seconds
        ssize_t cq_rc = fi_cq_read(impl_->cq_, &cq_entry, 1);
        
        if (cq_rc > 0) {
            // Successfully received a message
            size_t msg_len = strlen(buffer);
            buffer[buffer_size - 1] = '\0'; // Ensure null termination
            std::cout << "[Libfabric Client] Received: " << buffer << std::endl;
            
            dummy_event.is_done = true;
            dummy_event.error_code = 0;
            dummy_event.error_message = "";
            dummy_event.bytes_transferred = msg_len;
            dummy_event.transport_used = hshm::lbm::TransportType::RDMA;
            received = true;
        } else if (cq_rc < 0 && cq_rc != -EAGAIN) {
            // Real error
            dummy_event.is_done = false;
            dummy_event.error_code = cq_rc;
            dummy_event.error_message = fi_strerror(-cq_rc);
            dummy_event.bytes_transferred = 0;
            dummy_event.transport_used = hshm::lbm::TransportType::RDMA;
            break;
        }
        // For EAGAIN, sleep briefly and try again
        if (!received) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
    
    if (!received) {
        dummy_event.is_done = false;
        dummy_event.error_code = -ETIMEDOUT;
        dummy_event.error_message = "Receive timeout";
        dummy_event.bytes_transferred = 0;
        dummy_event.transport_used = hshm::lbm::TransportType::RDMA;
    }
    
    return &dummy_event;
}

void Client::ProcessCompletions() {
    if (!impl_->connected) return;
    
    // Process any pending completions
    struct fi_cq_entry cq_entry;
    ssize_t rc = fi_cq_read(impl_->cq_, &cq_entry, 1);
    if (rc > 0) {
        std::cout << "[Libfabric Client] Completion processed" << std::endl;
    }
}

} // namespace hshm::lbm::libfabric