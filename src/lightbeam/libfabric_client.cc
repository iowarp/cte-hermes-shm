#include "hermes_shm/lightbeam/libfabric/client.h"
#include <rdma/fabric.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_rma.h>
#include <rdma/fi_domain.h>
#include <iostream>
#include <cstring>
#include <thread>
#include <chrono>
#include <queue>
#include <unordered_map>
#include <atomic>
#include <algorithm>

namespace hshm::lbm::libfabric {

struct PendingOperation {
    std::unique_ptr<Event> event;
    OperationType op_type;
    void* buffer;
    size_t buffer_size;
    struct fi_context context;
    std::shared_ptr<MemoryRegion> mr; // Keep MR alive
    
    // Constructor to initialize context properly
    PendingOperation() {
        memset(&context, 0, sizeof(context));
    }
};

struct Client::Impl {
    struct fid_fabric* fabric = nullptr;
    struct fid_domain* domain = nullptr;
    struct fid_ep* endpoint = nullptr;
    struct fid_cq* cq = nullptr;
    struct fid_eq* eq = nullptr;
    bool connected = false;
    
    // Event management
    std::atomic<uint64_t> next_event_id{1};
    std::queue<std::unique_ptr<Event>> completed_events;
    std::unordered_map<uint64_t, std::unique_ptr<PendingOperation>> pending_ops;
    
    // Memory registration cache
    std::unordered_map<void*, std::shared_ptr<MemoryRegion>> mr_cache;
    
    uint64_t get_next_event_id() {
        return next_event_id.fetch_add(1);
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

Client::Client() : impl_(std::make_unique<Impl>()) {}
Client::~Client() {
    if (impl_) {
        impl_->connected = false;
        Disconnect("127.0.0.1:5556");
    }
}

std::shared_ptr<MemoryRegion> Client::RegisterMemory(void* addr, size_t length, uint64_t access_flags) {
    if (!impl_->domain) {
        std::cerr << "[Libfabric Client] Cannot register memory: domain not initialized" << std::endl;
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
    
    // Validate access flags and set appropriate defaults
    if (access_flags == 0) {
        // Default access for general purpose operations
        access_flags = FI_SEND | FI_RECV | FI_READ | FI_WRITE | FI_REMOTE_READ | FI_REMOTE_WRITE;
    }
    
    // Register memory with libfabric - use the actual libfabric fid_mr type
    struct fid_mr* mr_ptr = nullptr;
    int rc = fi_mr_reg(impl_->domain, addr, length, access_flags, 0, 0, 0, &mr_ptr, nullptr);
    if (rc != 0) {
        std::cerr << "[Libfabric Client] fi_mr_reg failed: " << fi_strerror(rc) 
                  << " (addr=" << addr << ", length=" << length << ", access=0x" 
                  << std::hex << access_flags << std::dec << ")" << std::endl;
        return nullptr;
    }
    
    // Store as void* to avoid type issues
    mr_region->mr = static_cast<void*>(mr_ptr);
    
    // Get memory descriptor and key
    mr_region->desc = fi_mr_desc(mr_ptr);
    mr_region->key = fi_mr_key(mr_ptr);
    
    std::cout << "[Libfabric Client] Registered memory region: addr=" << addr 
              << " length=" << length << " key=" << mr_region->key 
              << " access=0x" << std::hex << access_flags << std::dec << std::endl;
    
    // Cache the memory region
    impl_->mr_cache[addr] = mr_region;
    
    return mr_region;
}

void Client::Connect(const std::string& url, hshm::lbm::TransportType transport) {
    std::cout << "[Libfabric Client] Connecting to " << url << std::endl;
    
    auto [node, service] = parseUrl(url);
    std::cout << "[Libfabric Client] Parsed address: " << node << ":" << service << std::endl;
    
    // Try different providers in order of preference
    const char* providers[] = {"sockets", "tcp", "udp", nullptr};
    struct fi_info* info = nullptr;
    int rc = -1;
    
    for (int i = 0; providers[i] != nullptr; i++) {
        std::cout << "[Libfabric Client] Trying provider: " << providers[i] << std::endl;
        
        // Initialize fabric
        struct fi_info* hints = fi_allocinfo();
        hints->ep_attr->type = FI_EP_MSG;
        hints->domain_attr->threading = FI_THREAD_SAFE;
        hints->fabric_attr->prov_name = strdup(providers[i]);
        hints->caps = FI_MSG | FI_RMA | FI_READ | FI_WRITE | FI_REMOTE_READ | FI_REMOTE_WRITE;
        hints->mode = FI_CONTEXT; // Explicitly request FI_CONTEXT mode
        
        rc = fi_getinfo(FI_VERSION(1, 1), node.c_str(), service.c_str(), 0, hints, &info);
        if (rc == 0) {
            std::cout << "[Libfabric Client] Successfully found provider: " << providers[i] << std::endl;
            
            // Validate provider capabilities
            std::cout << "[Libfabric Client] Provider capabilities:" << std::endl;
            std::cout << "  - Max message size: " << info->ep_attr->max_msg_size << std::endl;
            std::cout << "  - Protocol: " << info->ep_attr->protocol << std::endl;
            std::cout << "  - Threading: " << info->domain_attr->threading << std::endl;
            std::cout << "  - Mode bits: 0x" << std::hex << info->mode << std::dec << std::endl;
            std::cout << "  - Capabilities: 0x" << std::hex << info->caps << std::dec << std::endl;
            
            // Check for required capabilities
            uint64_t required_caps = FI_MSG | FI_RMA;
            if ((info->caps & required_caps) != required_caps) {
                std::cerr << "[Libfabric Client] Provider missing required capabilities" << std::endl;
                fi_freeinfo(info);
                fi_freeinfo(hints);
                continue;
            }
            
            fi_freeinfo(hints);
            break;
        } else {
            std::cout << "[Libfabric Client] Provider " << providers[i] << " failed: " << fi_strerror(rc) << std::endl;
            fi_freeinfo(hints);
        }
    }
    
    if (rc != 0) {
        std::cerr << "[Libfabric Client] No suitable provider found. RDMA may not be available on this system." << std::endl;
        return;
    }
    
    // Create fabric
    rc = fi_fabric(info->fabric_attr, &impl_->fabric, nullptr);
    if (rc != 0) {
        std::cerr << "[Libfabric Client] fi_fabric failed: " << fi_strerror(rc) << std::endl;
        fi_freeinfo(info);
        return;
    }
    
    // Create domain
    rc = fi_domain(impl_->fabric, info, &impl_->domain, nullptr);
    if (rc != 0) {
        std::cerr << "[Libfabric Client] fi_domain failed: " << fi_strerror(rc) << std::endl;
        fi_close(&impl_->fabric->fid);
        fi_freeinfo(info);
        return;
    }
    
    // Create completion queue - use DATA format to get length info
    struct fi_cq_attr cq_attr = {0};
    cq_attr.size = 128;
    cq_attr.flags = 0; // Remove conflicting flags
    cq_attr.format = FI_CQ_FORMAT_DATA; // Use DATA format to get length and flags
    cq_attr.wait_obj = FI_WAIT_NONE; // Non-blocking CQ reads
    rc = fi_cq_open(impl_->domain, &cq_attr, &impl_->cq, nullptr);
    if (rc != 0) {
        std::cerr << "[Libfabric Client] fi_cq_open failed: " << fi_strerror(rc) << std::endl;
        fi_close(&impl_->domain->fid);
        fi_close(&impl_->fabric->fid);
        fi_freeinfo(info);
        return;
    }
    
    // Create event queue for connection management
    struct fi_eq_attr eq_attr = {0};
    eq_attr.size = 10;
    eq_attr.wait_obj = FI_WAIT_NONE; // Non-blocking EQ reads
    rc = fi_eq_open(impl_->fabric, &eq_attr, &impl_->eq, nullptr);
    if (rc != 0) {
        std::cerr << "[Libfabric Client] fi_eq_open failed: " << fi_strerror(rc) << std::endl;
        fi_close(&impl_->cq->fid);
        fi_close(&impl_->domain->fid);
        fi_close(&impl_->fabric->fid);
        fi_freeinfo(info);
        return;
    }
    
    // Create endpoint
    rc = fi_endpoint(impl_->domain, info, &impl_->endpoint, nullptr);
    if (rc != 0) {
        std::cerr << "[Libfabric Client] fi_endpoint failed: " << fi_strerror(rc) << std::endl;
        fi_close(&impl_->eq->fid);
        fi_close(&impl_->cq->fid);
        fi_close(&impl_->domain->fid);
        fi_close(&impl_->fabric->fid);
        fi_freeinfo(info);
        return;
    }
    
    // Bind CQ and EQ to endpoint with proper flags
    // Bind CQ for both transmit and receive operations
    rc = fi_ep_bind(impl_->endpoint, &impl_->cq->fid, FI_TRANSMIT | FI_RECV);
    if (rc != 0) {
        std::cerr << "[Libfabric Client] fi_ep_bind CQ failed: " << fi_strerror(rc) << std::endl;
        fi_close(&impl_->endpoint->fid);
        fi_close(&impl_->eq->fid);
        fi_close(&impl_->cq->fid);
        fi_close(&impl_->domain->fid);
        fi_close(&impl_->fabric->fid);
        fi_freeinfo(info);
        return;
    }
    
    // Bind EQ for connection management events
    rc = fi_ep_bind(impl_->endpoint, &impl_->eq->fid, 0);
    if (rc != 0) {
        std::cerr << "[Libfabric Client] fi_ep_bind EQ failed: " << fi_strerror(rc) << std::endl;
        fi_close(&impl_->endpoint->fid);
        fi_close(&impl_->eq->fid);
        fi_close(&impl_->cq->fid);
        fi_close(&impl_->domain->fid);
        fi_close(&impl_->fabric->fid);
        fi_freeinfo(info);
        return;
    }
    
    // Enable endpoint - must be done after binding
    rc = fi_enable(impl_->endpoint);
    if (rc != 0) {
        std::cerr << "[Libfabric Client] fi_enable failed: " << fi_strerror(rc) << std::endl;
        fi_close(&impl_->endpoint->fid);
        fi_close(&impl_->eq->fid);
        fi_close(&impl_->cq->fid);
        fi_close(&impl_->domain->fid);
        fi_close(&impl_->fabric->fid);
        fi_freeinfo(info);
        return;
    }
    
    // Connect to server using the address from info
    std::cout << "[Libfabric Client] Initiating connection..." << std::endl;
    rc = fi_connect(impl_->endpoint, info->dest_addr, nullptr, 0);
    if (rc != 0) {
        std::cerr << "[Libfabric Client] fi_connect failed: " << fi_strerror(rc) << std::endl;
        fi_close(&impl_->endpoint->fid);
        fi_close(&impl_->eq->fid);
        fi_close(&impl_->cq->fid);
        fi_close(&impl_->domain->fid);
        fi_close(&impl_->fabric->fid);
        fi_freeinfo(info);
        return;
    }
    
    // Use blocking read with timeout
    struct fi_eq_cm_entry entry;
    uint32_t event;
    ssize_t eq_rc = fi_eq_sread(impl_->eq, &event, &entry, sizeof(entry), 5000, 0);
    
    if (eq_rc > 0) {
        std::cout << "[Libfabric Client] Received event: " << event << std::endl;
        if (event == FI_CONNECTED) {
            impl_->connected = true;
            std::cout << "[Libfabric Client] Connected successfully with provider: " << info->fabric_attr->prov_name << std::endl;
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
    
    if (!impl_->connected) {
        std::cerr << "[Libfabric Client] Connection failed" << std::endl;
        if (impl_->endpoint) {
            fi_close(&impl_->endpoint->fid);
            impl_->endpoint = nullptr;
        }
    }
    
    fi_freeinfo(info);
}

void Client::Disconnect(const std::string& url) {
    if (!impl_->connected && !impl_->endpoint) return;
    
    std::cout << "[Libfabric Client] Disconnecting from " << url << std::endl;
    impl_->connected = false;
    
    // Clear memory registration cache
    impl_->mr_cache.clear();
    
    // Clear pending operations
    impl_->pending_ops.clear();
    
    if (impl_->endpoint) {
        fi_close(&impl_->endpoint->fid);
        impl_->endpoint = nullptr;
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

hshm::lbm::Bulk Client::Expose(const std::string& url, const char* data, size_t data_size, int flags) {
    std::cout << "[Libfabric Client] Exposing data of size " << data_size << std::endl;
    
    hshm::lbm::Bulk bulk;
    bulk.data = const_cast<char*>(data);
    bulk.size = data_size;
    bulk.target_url = url;
    bulk.preferred_transport = hshm::lbm::TransportType::RDMA;
    
    // Register memory for RDMA operations
    uint64_t access = FI_SEND | FI_RECV | FI_READ | FI_WRITE | FI_REMOTE_READ | FI_REMOTE_WRITE;
    bulk.local_mr = RegisterMemory(bulk.data, bulk.size, access);
    
    if (!bulk.local_mr) {
        std::cerr << "[Libfabric Client] Failed to register memory for RDMA" << std::endl;
    }
    
    return bulk;
}

std::unique_ptr<hshm::lbm::Event> Client::Send(const hshm::lbm::Bulk& bulk) {
    auto event = std::make_unique<Event>();
    event->event_id = impl_->get_next_event_id();
    event->operation_type = OperationType::SEND;
    event->transport_used = hshm::lbm::TransportType::RDMA;
    event->start_time = std::chrono::steady_clock::now();
    
    if (!impl_->connected || !impl_->endpoint) {
        event->is_done = true;
        event->error_code = -1;
        event->error_message = "Not connected";
        return event;
    }
    
    std::cout << "[Libfabric Client] Initiating async send: " << bulk.data << std::endl;
    
    // Create message with header
    auto total_size = bulk.total_size();
    auto message_buffer = std::make_unique<char[]>(total_size);
    
    // Fill header
    auto* header = reinterpret_cast<Bulk::MessageHeader*>(message_buffer.get());
    header->magic = 0xDEADBEEF;
    header->size = bulk.size;
    header->sequence = event->event_id;
    
    // Copy payload
    std::memcpy(message_buffer.get() + sizeof(Bulk::MessageHeader), bulk.data, bulk.size);
    
    // Create pending operation BEFORE using context
    auto pending_op = std::make_unique<PendingOperation>();
    pending_op->event = std::move(event);
    pending_op->op_type = OperationType::SEND;
    pending_op->buffer = message_buffer.release(); // Transfer ownership
    pending_op->buffer_size = total_size;
    pending_op->mr = bulk.local_mr;
    
    // Store event ID in context - CRITICAL: context must remain valid until completion
    // We'll use the pending_op address as a unique identifier
    pending_op->context.internal[0] = static_cast<void*>(pending_op.get());
    
    // Get memory descriptor
    void* desc = nullptr;
    if (bulk.local_mr) {
        desc = bulk.local_mr->desc;
    }
    
    // Store the event ID for later lookup
    uint64_t event_id = pending_op->event->event_id;
    
    // Initiate non-blocking send - context must remain valid!
    int rc = fi_send(impl_->endpoint, pending_op->buffer, pending_op->buffer_size, desc, FI_ADDR_UNSPEC, &pending_op->context);
    
    // Return a copy of the event
    auto event_copy = std::make_unique<Event>(*pending_op->event);
    
    if (rc == 0) {
        // Operation posted successfully - store pending_op to keep context alive
        impl_->pending_ops[event_id] = std::move(pending_op);
        std::cout << "[Libfabric Client] Send operation posted with event_id: " << event_id << std::endl;
    } else {
        // Immediate error
        event_copy->is_done = true;
        event_copy->error_code = rc;
        event_copy->error_message = fi_strerror(rc);
        delete[] static_cast<char*>(pending_op->buffer);
    }
    
    return event_copy;
}

std::unique_ptr<hshm::lbm::Event> Client::Recv(char* buffer, size_t buffer_size, const std::string& from_url) {
    auto event = std::make_unique<Event>();
    event->event_id = impl_->get_next_event_id();
    event->operation_type = OperationType::RECV;
    event->transport_used = hshm::lbm::TransportType::RDMA;
    event->start_time = std::chrono::steady_clock::now();
    
    if (!impl_->connected || !impl_->endpoint) {
        event->is_done = true;
        event->error_code = -1;
        event->error_message = "Not connected";
        return event;
    }
    
    if (!buffer || buffer_size == 0) {
        event->is_done = true;
        event->error_code = -1;
        event->error_message = "Invalid buffer";
        return event;
    }
    
    std::cout << "[Libfabric Client] Initiating async receive..." << std::endl;
    
    // Register receive buffer if needed
    auto mr = RegisterMemory(buffer, buffer_size, FI_RECV);
    
    // Create pending operation BEFORE using context
    auto pending_op = std::make_unique<PendingOperation>();
    pending_op->event = std::move(event);
    pending_op->op_type = OperationType::RECV;
    pending_op->buffer = buffer;
    pending_op->buffer_size = buffer_size;
    pending_op->mr = mr;
    
    // Store pending_op address in context for lookup
    pending_op->context.internal[0] = static_cast<void*>(pending_op.get());
    
    // Get memory descriptor
    void* desc = nullptr;
    if (mr) {
        desc = mr->desc;
    }
    
    // Store the event ID for later lookup
    uint64_t event_id = pending_op->event->event_id;
    
    // Post receive - context must remain valid!
    int rc = fi_recv(impl_->endpoint, buffer, buffer_size, desc, FI_ADDR_UNSPEC, &pending_op->context);
    
    // Return a copy of the event
    auto event_copy = std::make_unique<Event>(*pending_op->event);
    
    if (rc == 0) {
        // Operation posted successfully - store pending_op to keep context alive
        impl_->pending_ops[event_id] = std::move(pending_op);
        std::cout << "[Libfabric Client] Recv operation posted with event_id: " << event_id << std::endl;
    } else {
        // Immediate error
        event_copy->is_done = true;
        event_copy->error_code = rc;
        event_copy->error_message = fi_strerror(rc);
    }
    
    return event_copy;
}

std::unique_ptr<hshm::lbm::Event> Client::RmaWrite(const hshm::lbm::Bulk& local_bulk, const hshm::lbm::Bulk& remote_bulk) {
    auto event = std::make_unique<Event>();
    event->event_id = impl_->get_next_event_id();
    event->operation_type = OperationType::RMA_WRITE;
    event->transport_used = hshm::lbm::TransportType::RDMA;
    event->start_time = std::chrono::steady_clock::now();
    
    if (!impl_->connected || !impl_->endpoint) {
        event->is_done = true;
        event->error_code = -1;
        event->error_message = "Not connected";
        return event;
    }
    
    if (!local_bulk.local_mr || !remote_bulk.remote_mr) {
        event->is_done = true;
        event->error_code = -1;
        event->error_message = "Memory not registered for RMA";
        return event;
    }
    
    std::cout << "[Libfabric Client] Initiating RMA write..." << std::endl;
    
    // Create pending operation
    auto pending_op = std::make_unique<PendingOperation>();
    pending_op->event = std::move(event);
    pending_op->op_type = OperationType::RMA_WRITE;
    pending_op->buffer = local_bulk.data;
    pending_op->buffer_size = local_bulk.size;
    pending_op->mr = local_bulk.local_mr;
    
    // Store event ID in context
    pending_op->context.internal[0] = reinterpret_cast<void*>(pending_op->event->event_id);
    
    // Perform RMA write
    int rc = fi_write(impl_->endpoint, 
                      local_bulk.data, 
                      local_bulk.size,
                      local_bulk.local_mr->desc,
                      FI_ADDR_UNSPEC,
                      reinterpret_cast<uint64_t>(remote_bulk.remote_mr->addr),
                      remote_bulk.remote_mr->key,
                      &pending_op->context);
    
    // Return a copy of the event
    auto event_copy = std::make_unique<Event>(*pending_op->event);
    
    if (rc == 0) {
        uint64_t event_id = pending_op->event->event_id;
        impl_->pending_ops[event_id] = std::move(pending_op);
        std::cout << "[Libfabric Client] RMA write posted with event_id: " << event_id << std::endl;
    } else {
        event_copy->is_done = true;
        event_copy->error_code = rc;
        event_copy->error_message = fi_strerror(rc);
    }
    
    return event_copy;
}

void Client::ProcessCompletions(double timeout_msec) {
    if (!impl_->connected || !impl_->cq) return;
    
    struct fi_cq_data_entry cq_entry; // Use data entry format to get length
    ssize_t rc;
    
    // Process all available completions
    while ((rc = fi_cq_read(impl_->cq, &cq_entry, 1)) > 0) {
        // Extract pending operation pointer from context
        // The context is a fi_context structure, and we stored the PendingOperation pointer in internal[0]
        struct fi_context* context = static_cast<struct fi_context*>(cq_entry.op_context);
        PendingOperation* pending_op_ptr = static_cast<PendingOperation*>(context->internal[0]);
        
        // Find the corresponding pending operation
        auto it = std::find_if(impl_->pending_ops.begin(), impl_->pending_ops.end(),
            [pending_op_ptr](const auto& pair) {
                return pair.second.get() == pending_op_ptr;
            });
        
        if (it != impl_->pending_ops.end()) {
            auto& pending_op = it->second;
            
            // Mark event as completed
            pending_op->event->is_done = true;
            pending_op->event->error_code = 0;
            pending_op->event->bytes_transferred = cq_entry.len;
            
            std::cout << "[Libfabric Client] Operation " << it->first << " completed successfully, "
                     << cq_entry.len << " bytes transferred" << std::endl;
            
            // For SEND operations, clean up allocated buffer
            if (pending_op->op_type == OperationType::SEND) {
                delete[] static_cast<char*>(pending_op->buffer);
            }
            
            impl_->pending_ops.erase(it);
        } else {
            std::cerr << "[Libfabric Client] Warning: Received completion for unknown operation" << std::endl;
        }
    }
    
    // Handle completion queue errors - FI_EAGAIN is expected when no completions available
    if (rc < 0 && rc != -FI_EAGAIN) {
        std::cerr << "[Libfabric Client] CQ read error: " << fi_strerror(-rc) << std::endl;
        
        // Check for error entries in the CQ
        struct fi_cq_err_entry error_entry;
        ssize_t error_rc = fi_cq_readerr(impl_->cq, &error_entry, 0);
        if (error_rc > 0) {
            struct fi_context* context = static_cast<struct fi_context*>(error_entry.op_context);
            PendingOperation* pending_op_ptr = static_cast<PendingOperation*>(context->internal[0]);
            
            // Find the corresponding pending operation
            auto it = std::find_if(impl_->pending_ops.begin(), impl_->pending_ops.end(),
                [pending_op_ptr](const auto& pair) {
                    return pair.second.get() == pending_op_ptr;
                });
            
            if (it != impl_->pending_ops.end()) {
                auto& pending_op = it->second;
                
                // Mark event as failed
                pending_op->event->is_done = true;
                pending_op->event->error_code = error_entry.err;
                pending_op->event->error_message = fi_strerror(error_entry.err);
                
                std::cerr << "[Libfabric Client] Operation " << it->first << " failed: " 
                         << pending_op->event->error_message << " (provider errno: " 
                         << error_entry.prov_errno << ")" << std::endl;
                
                // Clean up if SEND operation
                if (pending_op->op_type == OperationType::SEND) {
                    delete[] static_cast<char*>(pending_op->buffer);
                }
                
                impl_->pending_ops.erase(it);
            } else {
                std::cerr << "[Libfabric Client] Warning: Error completion for unknown operation" << std::endl;
            }
        } else if (error_rc < 0) {
            std::cerr << "[Libfabric Client] Error reading error queue: " << fi_strerror(-error_rc) << std::endl;
        }
    }
    
    // Check for timeouts
    auto now = std::chrono::steady_clock::now();
    for (auto it = impl_->pending_ops.begin(); it != impl_->pending_ops.end();) {
        if (it->second->event->has_timed_out()) {
            auto& pending_op = it->second;
            
            // Mark as timed out
            pending_op->event->is_done = true;
            pending_op->event->error_code = -ETIMEDOUT;
            pending_op->event->error_message = "Operation timed out";
            
            std::cerr << "[Libfabric Client] Operation " << it->first << " timed out" << std::endl;
            
            // Clean up if SEND operation
            if (pending_op->op_type == OperationType::SEND) {
                delete[] static_cast<char*>(pending_op->buffer);
            }
            
            it = impl_->pending_ops.erase(it);
        } else {
            ++it;
        }
    }
}

} // namespace hshm::lbm::libfabric