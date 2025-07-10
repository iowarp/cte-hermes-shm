#pragma once
#include <rdma/fabric.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_cm.h>
#include <memory>
#include <string>
#include "hermes_shm/lightbeam/types.h"

namespace hshm::lbm::libfabric {

/**
 * @brief Common libfabric resources and operations
 */
class LibfabricCommon {
public:
    LibfabricCommon();
    virtual ~LibfabricCommon();

public:
    /**
     * @brief Initialize libfabric resources (fabric, domain, queues)
     * @param node Address node (e.g., "127.0.0.1")
     * @param service Port/service (e.g., "5556") 
     * @param flags Flags for fi_getinfo (e.g., FI_SOURCE for server, 0 for client)
     * @return 0 on success, negative on failure
     */
    int initializeResources(const std::string& node, const std::string& service, uint64_t flags);

    /**
     * @brief Clean up all libfabric resources
     */
    void cleanup();

    /**
     * @brief Check if resources are initialized
     */
    bool isInitialized() const { return fabric_ != nullptr; }

    // Common libfabric resources
    struct fid_fabric* fabric_ = nullptr;
    struct fid_domain* domain_ = nullptr; 
    struct fid_cq* cq_ = nullptr;
    struct fid_eq* eq_ = nullptr;
    struct fi_info* info_ = nullptr;

    // Common buffers
    char recv_buffer_[1024];
    char send_buffer_[1024];
    struct fi_context recv_ctx_;
    struct fi_context send_ctx_;

private:
    bool initialized_ = false;
};

} // namespace hshm::lbm::libfabric 