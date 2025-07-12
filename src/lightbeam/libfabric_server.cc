#include "hermes_shm/lightbeam/libfabric/server.h"
#include <rdma/fabric.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_rma.h>
#include <rdma/fi_domain.h>
#include <iostream>
#include <cstring>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <queue>
#include <unordered_map>
#include <memory>
#include <atomic>
#include <array>
#include <algorithm>

namespace hshm::lbm::libfabric {

struct ServerOperation {
    OperationType op_type;
    void* buffer;
    size_t buffer_size;
    struct fi_context context;
    std::shared_ptr<MemoryRegion> mr;
    uint64_t operation_id;
};

struct Server::Impl {
    struct fid_fabric* fabric = nullptr;
    struct fid_domain* domain = nullptr;
    struct fid_ep* endpoint = nullptr;
    struct fid_pep* pep = nullptr;
    struct fid_cq* cq = nullptr;
    struct fid_eq* eq = nullptr;
    bool running = false;
    
    // Operation management
    std::atomic<uint64_t> next_operation_id{1};
    std::unordered_map<uint64_t, std::unique_ptr<ServerOperation>> active_operations;
    
    // Receive buffer pool
    static constexpr size_t RECV_BUFFER_SIZE = 64 * 1024; // 64KB
    static constexpr size_t NUM_RECV_BUFFERS = 16;
    std::array<std::unique_ptr<char[]>, NUM_RECV_BUFFERS> recv_buffers;
    std::array<std::shared_ptr<MemoryRegion>, NUM_RECV_BUFFERS> recv_mrs;
    std::queue<size_t> available_recv_buffers;
    
    // Memory registration cache
    std::unordered_map<void*, std::shared_ptr<MemoryRegion>> mr_cache;
    
    uint64_t get_next_operation_id() {
        return next_operation_id.fetch_add(1);
    }
};

// Helper function to parse URL
static std::pair<std::string, std::string> parseUrl(const std::string& url) {
    size_t protocol_end = url.find("://");
    if (protocol_end == std::string::npos) {
        return {"127.0.0.1", "5556"};
    }
    
    std::string address_port = url.substr(protocol_end + 3);
    size_t colon_pos = address_port.find(':');
    if (colon_pos == std::string::npos) {
        return {address_port, "5556"};
    }
    
    return {address_port.substr(0, colon_pos), address_port.substr(colon_pos + 1)};
}

Server::Server() : impl_(std::make_unique<Impl>()) {}
Server::~Server() {
    if (impl_) {
        impl_->running = false;
        Stop();
    }
}

std::shared_ptr<MemoryRegion> Server::RegisterMemory(void* addr, size_t length, uint64_t access_flags) {
    if (!impl_->domain) {
        std::cerr << "[Libfabric Server] Cannot register memory: domain not initialized" << std::endl;
        return nullptr;
    }
    
    // Check cache first
    auto it = impl_->mr_cache.find(addr);
    if (it != impl_->mr_cache.end()) {
        return it->second;
    }
    
    auto mr_region = std::make_shared<MemoryRegion>();
    mr_region->addr = addr;
    mr_region->length = length;
    
    // Register memory with libfabric - use the actual libfabric fid_mr type
    struct fid_mr* mr_ptr = nullptr;
    int rc = fi_mr_reg(impl_->domain, addr, length, access_flags, 0, 0, 0, &mr_ptr, nullptr);
    if (rc != 0) {
        std::cerr << "[Libfabric Server] fi_mr_reg failed: " << fi_strerror(rc) << std::endl;
        return nullptr;
    }
    
    // Store as void* to avoid type issues
    mr_region->mr = static_cast<void*>(mr_ptr);
    
    // Get memory descriptor and key
    mr_region->desc = fi_mr_desc(mr_ptr);
    mr_region->key = fi_mr_key(mr_ptr);
    
    std::cout << "[Libfabric Server] Registered memory region: addr=" << addr 
              << " length=" << length << " key=" << mr_region->key << std::endl;
    
    // Cache the memory region
    impl_->mr_cache[addr] = mr_region;
    
    return mr_region;
}

void Server::InitializeReceiveBuffers() {
    // Initialize receive buffer pool
    for (size_t i = 0; i < Impl::NUM_RECV_BUFFERS; ++i) {
        impl_->recv_buffers[i] = std::make_unique<char[]>(Impl::RECV_BUFFER_SIZE);
        
        // Register each buffer
        uint64_t access = FI_RECV | FI_SEND | FI_READ | FI_WRITE | FI_REMOTE_READ | FI_REMOTE_WRITE;
        impl_->recv_mrs[i] = RegisterMemory(impl_->recv_buffers[i].get(), Impl::RECV_BUFFER_SIZE, access);
        
        if (impl_->recv_mrs[i]) {
            impl_->available_recv_buffers.push(i);
        }
    }
    
    std::cout << "[Libfabric Server] Initialized " << impl_->available_recv_buffers.size() 
              << " receive buffers" << std::endl;
}

void Server::PostReceiveOperations() {
    // Post multiple receive operations
    while (!impl_->available_recv_buffers.empty() && impl_->endpoint) {
        size_t buffer_idx = impl_->available_recv_buffers.front();
        impl_->available_recv_buffers.pop();
        
        auto operation = std::make_unique<ServerOperation>();
        operation->op_type = OperationType::RECV;
        operation->buffer = impl_->recv_buffers[buffer_idx].get();
        operation->buffer_size = Impl::RECV_BUFFER_SIZE;
        operation->mr = impl_->recv_mrs[buffer_idx];
        operation->operation_id = impl_->get_next_operation_id();
        
        // Store operation pointer in context for lookup
        operation->context.internal[0] = static_cast<void*>(operation.get());
        
        void* desc = operation->mr ? operation->mr->desc : nullptr;
        
        int rc = fi_recv(impl_->endpoint, operation->buffer, operation->buffer_size, 
                        desc, FI_ADDR_UNSPEC, &operation->context);
        
        if (rc == 0) {
            uint64_t op_id = operation->operation_id;
            impl_->active_operations[op_id] = std::move(operation);
            std::cout << "[Libfabric Server] Posted receive operation " << op_id 
                     << " with buffer " << buffer_idx << std::endl;
        } else {
            std::cerr << "[Libfabric Server] Failed to post receive: " << fi_strerror(rc) << std::endl;
            impl_->available_recv_buffers.push(buffer_idx); // Return buffer to pool
            break;
        }
    }
}

void Server::StartServer(const std::string& url, hshm::lbm::TransportType transport) {
    std::cout << "[Libfabric Server] Starting server on " << url << std::endl;
    
    auto [node, service] = parseUrl(url);
    std::cout << "[Libfabric Server] Parsed address: " << node << ":" << service << std::endl;
    
    // Try different providers in order of preference
    const char* providers[] = {"sockets", "tcp", "udp", nullptr};
    struct fi_info* info = nullptr;
    int rc = -1;
    
    for (int i = 0; providers[i] != nullptr; i++) {
        std::cout << "[Libfabric Server] Trying provider: " << providers[i] << std::endl;
        
        // Initialize fabric
        struct fi_info* hints = fi_allocinfo();
        hints->ep_attr->type = FI_EP_MSG;
        hints->domain_attr->threading = FI_THREAD_SAFE;
        hints->fabric_attr->prov_name = strdup(providers[i]);
        hints->caps = FI_MSG | FI_RMA | FI_READ | FI_WRITE | FI_REMOTE_READ | FI_REMOTE_WRITE;
        
        rc = fi_getinfo(FI_VERSION(1, 1), node.c_str(), service.c_str(), FI_SOURCE, hints, &info);
        if (rc == 0) {
            std::cout << "[Libfabric Server] Successfully found provider: " << providers[i] << std::endl;
            fi_freeinfo(hints);
            break;
        } else {
            std::cout << "[Libfabric Server] Provider " << providers[i] << " failed: " << fi_strerror(rc) << std::endl;
            fi_freeinfo(hints);
        }
    }
    
    if (rc != 0) {
        std::cerr << "[Libfabric Server] No suitable provider found. RDMA may not be available on this system." << std::endl;
        return;
    }
    
    // Create fabric
    rc = fi_fabric(info->fabric_attr, &impl_->fabric, nullptr);
    if (rc != 0) {
        std::cerr << "[Libfabric Server] fi_fabric failed: " << fi_strerror(rc) << std::endl;
        fi_freeinfo(info);
        return;
    }
    
    // Create domain
    rc = fi_domain(impl_->fabric, info, &impl_->domain, nullptr);
    if (rc != 0) {
        std::cerr << "[Libfabric Server] fi_domain failed: " << fi_strerror(rc) << std::endl;
        fi_close(&impl_->fabric->fid);
        fi_freeinfo(info);
        return;
    }
    
    // Create completion queue - use DATA format to get length info
    struct fi_cq_attr cq_attr = {0};
    cq_attr.size = 128;
    cq_attr.flags = FI_COMPLETION | FI_SELECTIVE_COMPLETION;
    cq_attr.format = FI_CQ_FORMAT_DATA; // Use DATA format to get length
    rc = fi_cq_open(impl_->domain, &cq_attr, &impl_->cq, nullptr);
    if (rc != 0) {
        std::cerr << "[Libfabric Server] fi_cq_open failed: " << fi_strerror(rc) << std::endl;
        fi_close(&impl_->domain->fid);
        fi_close(&impl_->fabric->fid);
        fi_freeinfo(info);
        return;
    }
    
    // Create event queue
    struct fi_eq_attr eq_attr = {0};
    eq_attr.size = 10;
    rc = fi_eq_open(impl_->fabric, &eq_attr, &impl_->eq, nullptr);
    if (rc != 0) {
        std::cerr << "[Libfabric Server] fi_eq_open failed: " << fi_strerror(rc) << std::endl;
        fi_close(&impl_->cq->fid);
        fi_close(&impl_->domain->fid);
        fi_close(&impl_->fabric->fid);
        fi_freeinfo(info);
        return;
    }
    
    // Create passive endpoint
    rc = fi_passive_ep(impl_->fabric, info, &impl_->pep, nullptr);
    if (rc != 0) {
        std::cerr << "[Libfabric Server] fi_passive_ep failed: " << fi_strerror(rc) << std::endl;
        fi_close(&impl_->eq->fid);
        fi_close(&impl_->cq->fid);
        fi_close(&impl_->domain->fid);
        fi_close(&impl_->fabric->fid);
        fi_freeinfo(info);
        return;
    }
    
    // Bind EQ to PEP
    rc = fi_pep_bind(impl_->pep, &impl_->eq->fid, 0);
    if (rc != 0) {
        std::cerr << "[Libfabric Server] fi_pep_bind failed: " << fi_strerror(rc) << std::endl;
        fi_close(&impl_->pep->fid);
        fi_close(&impl_->eq->fid);
        fi_close(&impl_->cq->fid);
        fi_close(&impl_->domain->fid);
        fi_close(&impl_->fabric->fid);
        fi_freeinfo(info);
        return;
    }
    
    // Listen for connections
    std::cout << "[Libfabric Server] About to call fi_listen" << std::endl;
    rc = fi_listen(impl_->pep);
    if (rc != 0) {
        std::cerr << "[Libfabric Server] fi_listen failed: " << fi_strerror(rc) << std::endl;
        fi_close(&impl_->pep->fid);
        fi_close(&impl_->eq->fid);
        fi_close(&impl_->cq->fid);
        fi_close(&impl_->domain->fid);
        fi_close(&impl_->fabric->fid);
        fi_freeinfo(info);
        return;
    }
    std::cout << "[Libfabric Server] fi_listen succeeded" << std::endl;
    
    // Initialize receive buffers
    InitializeReceiveBuffers();
    
    impl_->running = true;
    std::cout << "[Libfabric Server] Server started successfully with provider: " << info->fabric_attr->prov_name << std::endl;
    
    fi_freeinfo(info);
}

void Server::Stop() {
    if (!impl_->running) return;
    
    std::cout << "[Libfabric Server] Stopping server" << std::endl;
    impl_->running = false;
    
    // Clear memory registration cache
    impl_->mr_cache.clear();
    
    // Clear active operations
    impl_->active_operations.clear();
    
    // Close resources in reverse order of creation
    if (impl_->endpoint) {
        fi_close(&impl_->endpoint->fid);
        impl_->endpoint = nullptr;
    }
    if (impl_->pep) {
        fi_close(&impl_->pep->fid);
        impl_->pep = nullptr;
    }
    if (impl_->eq) {
        fi_close(&impl_->eq->fid);
        impl_->eq = nullptr;
    }
    if (impl_->cq) {
        fi_close(&impl_->cq->fid);
        impl_->cq = nullptr;
    }
    if (impl_->domain) {
        fi_close(&impl_->domain->fid);
        impl_->domain = nullptr;
    }
    if (impl_->fabric) {
        fi_close(&impl_->fabric->fid);
        impl_->fabric = nullptr;
    }
}

void Server::ProcessMessages() {
    if (!impl_->running) return;
    
    // Check for connection events
    struct fi_eq_cm_entry entry;
    uint32_t event;
    ssize_t rc = fi_eq_read(impl_->eq, &event, &entry, sizeof(entry), 0);
    
    if (rc > 0) {
        std::cout << "[Libfabric Server] Event received: " << event << std::endl;
        
        if (event == FI_CONNREQ) {
            std::cout << "[Libfabric Server] Connection request received" << std::endl;
            
            // Create endpoint using the connection request info
            rc = fi_endpoint(impl_->domain, entry.info, &impl_->endpoint, nullptr);
            if (rc == 0) {
                std::cout << "[Libfabric Server] Endpoint created successfully" << std::endl;
                
                // Bind endpoint to completion and event queues
                rc = fi_ep_bind(impl_->endpoint, &impl_->cq->fid, FI_TRANSMIT | FI_RECV);
                if (rc != 0) {
                    std::cerr << "[Libfabric Server] fi_ep_bind CQ failed: " << fi_strerror(rc) << std::endl;
                }
                
                rc = fi_ep_bind(impl_->endpoint, &impl_->eq->fid, 0);
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
                    
                    // Post initial receive operations
                    PostReceiveOperations();
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
        // Only log real errors occasionally
        static int error_count = 0;
        if ((++error_count % 100) == 0) {
            std::cout << "[Libfabric Server] fi_eq_read error: " << fi_strerror(-rc) << std::endl;
        }
    }
    
    // Process completions
    ProcessCompletions();
}

void Server::ProcessCompletions() {
    if (!impl_->cq) return;
    
    struct fi_cq_data_entry cq_entry; // Use data entry format to get length
    ssize_t rc;
    
    // Process all available completions
    while ((rc = fi_cq_read(impl_->cq, &cq_entry, 1)) > 0) {
        // Extract operation pointer from context
        struct fi_context* context = static_cast<struct fi_context*>(cq_entry.op_context);
        ServerOperation* operation_ptr = static_cast<ServerOperation*>(context->internal[0]);
        
        // Find the corresponding operation
        auto it = std::find_if(impl_->active_operations.begin(), impl_->active_operations.end(),
            [operation_ptr](const auto& pair) {
                return pair.second.get() == operation_ptr;
            });
        
        if (it != impl_->active_operations.end()) {
            auto& operation = it->second;
            
            if (operation->op_type == OperationType::RECV) {
                std::cout << "[Libfabric Server] Received message: " << cq_entry.len << " bytes" << std::endl;
                
                // Parse message header
                if (cq_entry.len >= sizeof(Bulk::MessageHeader)) {
                    auto* header = reinterpret_cast<Bulk::MessageHeader*>(operation->buffer);
                    
                    if (header->magic == 0xDEADBEEF) {
                        std::cout << "[Libfabric Server] Valid message received: size=" << header->size 
                                 << " sequence=" << header->sequence << std::endl;
                        
                        // Extract payload
                        char* payload = static_cast<char*>(operation->buffer) + sizeof(Bulk::MessageHeader);
                        size_t payload_size = std::min(static_cast<size_t>(header->size), 
                                                      cq_entry.len - sizeof(Bulk::MessageHeader));
                        
                        // Echo back the payload (for testing)
                        SendEcho(payload, payload_size, header->sequence);
                    }
                }
                
                // For receive operations, we need to extract the buffer index from the operation
                // Find which buffer this operation was using
                for (size_t i = 0; i < Impl::NUM_RECV_BUFFERS; ++i) {
                    if (impl_->recv_buffers[i].get() == operation->buffer) {
                        impl_->available_recv_buffers.push(i);
                        break;
                    }
                }
                
                impl_->active_operations.erase(it);
                
                // Post a new receive operation
                PostReceiveOperations();
            } else if (operation->op_type == OperationType::SEND) {
                std::cout << "[Libfabric Server] Send operation " << operation->operation_id 
                         << " completed: " << cq_entry.len << " bytes" << std::endl;
                
                // Clean up the dynamically allocated echo buffer
                delete[] static_cast<char*>(operation->buffer);
                impl_->active_operations.erase(it);
            } else {
                std::cout << "[Libfabric Server] Operation " << operation->operation_id << " completed: " 
                         << cq_entry.len << " bytes" << std::endl;
                impl_->active_operations.erase(it);
            }
        } else {
            std::cerr << "[Libfabric Server] Warning: Received completion for unknown operation" << std::endl;
        }
    }
    
    // Handle errors
    if (rc < 0 && rc != -EAGAIN) {
        struct fi_cq_err_entry error_entry;
        ssize_t error_rc = fi_cq_readerr(impl_->cq, &error_entry, 0);
        if (error_rc > 0) {
            struct fi_context* context = static_cast<struct fi_context*>(error_entry.op_context);
            ServerOperation* operation_ptr = static_cast<ServerOperation*>(context->internal[0]);
            
            // Find the corresponding operation
            auto it = std::find_if(impl_->active_operations.begin(), impl_->active_operations.end(),
                [operation_ptr](const auto& pair) {
                    return pair.second.get() == operation_ptr;
                });
            
            if (it != impl_->active_operations.end()) {
                std::cerr << "[Libfabric Server] Operation " << it->second->operation_id << " failed: " 
                         << fi_strerror(error_entry.err) << std::endl;
                
                // Clean up failed operation
                if (it->second->op_type == OperationType::SEND) {
                    delete[] static_cast<char*>(it->second->buffer);
                }
                impl_->active_operations.erase(it);
            } else {
                std::cerr << "[Libfabric Server] Error completion for unknown operation" << std::endl;
            }
        }
    }
}

void Server::SendEcho(const char* data, size_t size, uint64_t sequence) {
    if (!impl_->endpoint) {
        std::cerr << "[Libfabric Server] Cannot send echo: endpoint not available" << std::endl;
        return;
    }
    
    // Allocate a dedicated buffer for the echo response (don't reuse receive buffers)
    size_t total_size = sizeof(Bulk::MessageHeader) + size;
    auto echo_buffer = std::make_unique<char[]>(total_size);
    
    // Create echo message with header
    auto* header = reinterpret_cast<Bulk::MessageHeader*>(echo_buffer.get());
    header->magic = 0xDEADBEEF;
    header->size = size;
    header->sequence = sequence;
    
    // Copy the data
    std::memcpy(echo_buffer.get() + sizeof(Bulk::MessageHeader), data, size);
    
    // Register memory for the echo buffer
    auto mr = RegisterMemory(echo_buffer.get(), total_size, FI_SEND);
    if (!mr) {
        std::cerr << "[Libfabric Server] Failed to register echo buffer memory" << std::endl;
        return;
    }
    
    // Create send operation
    auto operation = std::make_unique<ServerOperation>();
    operation->op_type = OperationType::SEND;
    operation->buffer = echo_buffer.release(); // Transfer ownership
    operation->buffer_size = total_size;
    operation->mr = mr;
    operation->operation_id = impl_->get_next_operation_id();
    
    // Store operation pointer in context for lookup
    operation->context.internal[0] = static_cast<void*>(operation.get());
    
    void* desc = mr ? mr->desc : nullptr;
    
    int rc = fi_send(impl_->endpoint, operation->buffer, total_size, desc, FI_ADDR_UNSPEC, &operation->context);
    
    if (rc == 0) {
        uint64_t op_id = operation->operation_id;
        impl_->active_operations[op_id] = std::move(operation);
        std::cout << "[Libfabric Server] Echoed back " << size << " bytes (sequence: " << sequence << ")" << std::endl;
    } else {
        std::cerr << "[Libfabric Server] Failed to send echo: " << fi_strerror(rc) << std::endl;
        delete[] static_cast<char*>(operation->buffer);
    }
}

bool Server::IsRunning() const {
    return impl_->running;
}

} // namespace hshm::lbm::libfabric