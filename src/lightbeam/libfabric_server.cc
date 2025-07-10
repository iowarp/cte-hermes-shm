#include "hermes_shm/lightbeam/libfabric/server.h"
#include "hermes_shm/lightbeam/libfabric/common.h"
#include "hermes_shm/lightbeam/utils.h"
#include <rdma/fabric.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_cm.h>
#include <iostream>
#include <cstring>

namespace hshm::lbm::libfabric {

struct Server::Impl : public LibfabricCommon {
    struct fid_ep* endpoint = nullptr;
    struct fid_pep* pep = nullptr;
    bool running = false;
};

Server::Server() : impl_(std::make_unique<Impl>()) {}
Server::~Server() {
    if (impl_) {
        impl_->running = false;
        Stop();
    }
}

void Server::StartServer(const std::string& url, hshm::lbm::TransportType transport) {
    std::cout << "[Libfabric Server] Starting server on " << url << std::endl;
    
    auto [node, service] = utils::parseUrl(url);
    std::cout << "[Libfabric Server] Parsed address: " << node << ":" << service << std::endl;
    
    // Initialize common libfabric resources
    int rc = impl_->initializeResources(node, service, FI_SOURCE);
    if (rc != 0) {
        std::cerr << "[Libfabric Server] Failed to initialize libfabric resources" << std::endl;
        return;
    }
    
    // Create passive endpoint
    rc = fi_passive_ep(impl_->fabric_, impl_->info_, &impl_->pep, nullptr);
    if (rc != 0) {
        std::cerr << "[Libfabric Server] fi_passive_ep failed: " << fi_strerror(rc) << std::endl;
        impl_->cleanup();
        return;
    }
    
    // Bind EQ to PEP
    rc = fi_pep_bind(impl_->pep, &impl_->eq_->fid, 0);
    if (rc != 0) {
        std::cerr << "[Libfabric Server] fi_pep_bind failed: " << fi_strerror(rc) << std::endl;
        fi_close(&impl_->pep->fid);
        impl_->pep = nullptr;
        impl_->cleanup();
        return;
    }
    
    // Listen for connections
    std::cout << "[Libfabric Server] About to call fi_listen" << std::endl;
    rc = fi_listen(impl_->pep);
    if (rc != 0) {
        std::cerr << "[Libfabric Server] fi_listen failed: " << fi_strerror(rc) << std::endl;
        fi_close(&impl_->pep->fid);
        impl_->pep = nullptr;
        impl_->cleanup();
        return;
    }
    std::cout << "[Libfabric Server] fi_listen succeeded" << std::endl;
    
    impl_->running = true;
    std::cout << "[Libfabric Server] Server started successfully with provider: " << impl_->info_->fabric_attr->prov_name << std::endl;
}

void Server::Stop() {
    if (!impl_->running) return;
    
    std::cout << "[Libfabric Server] Stopping server" << std::endl;
    impl_->running = false;
    
    // Close server-specific resources
    if (impl_->endpoint) {
        fi_close(&impl_->endpoint->fid);
        impl_->endpoint = nullptr;
    }
    if (impl_->pep) {
        fi_close(&impl_->pep->fid);
        impl_->pep = nullptr;
    }
    
    // Cleanup common resources
    impl_->cleanup();
}

void Server::ProcessMessages() {
    if (!impl_->running) return;
    
    static int call_count = 0;
    call_count++;
    if (call_count % 10 == 0) {
        std::cout << "[Libfabric Server] ProcessMessages called " << call_count << " times" << std::endl;
    }
    
    // Check for connection events
    struct fi_eq_cm_entry entry;
    uint32_t event;
    ssize_t rc = fi_eq_read(impl_->eq_, &event, &entry, sizeof(entry), 0);
    
    if (rc > 0) {
        std::cout << "[Libfabric Server] Event received: " << event << std::endl;
        
        if (event == FI_CONNREQ) {
            std::cout << "[Libfabric Server] Connection request received" << std::endl;
            
            // Create endpoint using the connection request info
            rc = fi_endpoint(impl_->domain_, entry.info, &impl_->endpoint, nullptr);
            if (rc == 0) {
                std::cout << "[Libfabric Server] Endpoint created successfully" << std::endl;
                
                // Bind endpoint to completion and event queues
                rc = fi_ep_bind(impl_->endpoint, &impl_->cq_->fid, FI_TRANSMIT | FI_RECV);
                if (rc != 0) {
                    std::cerr << "[Libfabric Server] fi_ep_bind CQ failed: " << fi_strerror(rc) << std::endl;
                }
                
                rc = fi_ep_bind(impl_->endpoint, &impl_->eq_->fid, 0);
                if (rc != 0) {
                    std::cerr << "[Libfabric Server] fi_ep_bind EQ failed: " << fi_strerror(rc) << std::endl;
                }
                
                // Enable the endpoint
                rc = fi_enable(impl_->endpoint);
                if (rc != 0) {
                    std::cerr << "[Libfabric Server] fi_enable failed: " << fi_strerror(rc) << std::endl;
                }
                
                // Accept the connection
                rc = fi_accept(impl_->endpoint, nullptr, 0);
                if (rc == 0) {
                    std::cout << "[Libfabric Server] Connection accepted" << std::endl;
                    
                    // Post receive buffer
                    rc = fi_recv(impl_->endpoint, impl_->recv_buffer_, sizeof(impl_->recv_buffer_), 
                               nullptr, FI_ADDR_UNSPEC, &impl_->recv_ctx_);
                    if (rc != 0) {
                        std::cerr << "[Libfabric Server] fi_recv failed: " << fi_strerror(rc) << std::endl;
                    }
                } else {
                    std::cerr << "[Libfabric Server] fi_accept failed: " << fi_strerror(rc) << std::endl;
                }
            } else {
                std::cerr << "[Libfabric Server] fi_endpoint failed: " << fi_strerror(rc) << std::endl;
            }
        } else if (event == FI_CONNECTED) {
            std::cout << "[Libfabric Server] Connection established" << std::endl;
        } else if (event == FI_SHUTDOWN) {
            std::cout << "[Libfabric Server] Connection shutdown" << std::endl;
        } else {
            std::cout << "[Libfabric Server] Unknown event: " << event << std::endl;
        }
    } else if (rc < 0 && rc != -EAGAIN) {
        // Don't log every EAGAIN, only real errors
        if (call_count % 100 == 0) { // Only log occasionally
            std::cout << "[Libfabric Server] fi_eq_read error: " << fi_strerror(-rc) << std::endl;
        }
    }
    
    // Check for completions only if we have an endpoint
    if (impl_->endpoint) {
        struct fi_cq_entry cq_entry;
        rc = fi_cq_read(impl_->cq_, &cq_entry, 1);
        if (rc > 0) {
            // Received a message - echo it back
            size_t msg_len = strlen(impl_->recv_buffer_);
            if (msg_len > 0) {
                impl_->recv_buffer_[sizeof(impl_->recv_buffer_) - 1] = '\0'; // Ensure null termination
                std::cout << "[Libfabric Server] Received: " << impl_->recv_buffer_ << std::endl;
                
                // Echo back
                strcpy(impl_->send_buffer_, impl_->recv_buffer_);
                fi_send(impl_->endpoint, impl_->send_buffer_, strlen(impl_->send_buffer_), 
                       nullptr, FI_ADDR_UNSPEC, &impl_->send_ctx_);
                std::cout << "[Libfabric Server] Echoed back: " << impl_->send_buffer_ << std::endl;
                
                // Clear the receive buffer and post next receive
                memset(impl_->recv_buffer_, 0, sizeof(impl_->recv_buffer_));
                fi_recv(impl_->endpoint, impl_->recv_buffer_, sizeof(impl_->recv_buffer_), 
                       nullptr, FI_ADDR_UNSPEC, &impl_->recv_ctx_);
            }
        }
    }
}

bool Server::IsRunning() const {
    return impl_->running;
}

} // namespace hshm::lbm::libfabric