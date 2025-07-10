#include "hermes_shm/lightbeam/libfabric/common.h"
#include "hermes_shm/lightbeam/utils.h"
#include <iostream>
#include <cstring>

namespace hshm::lbm::libfabric {

LibfabricCommon::LibfabricCommon() {
    memset(recv_buffer_, 0, sizeof(recv_buffer_));
    memset(send_buffer_, 0, sizeof(send_buffer_));
}

LibfabricCommon::~LibfabricCommon() {
    cleanup();
}

int LibfabricCommon::initializeResources(const std::string& node, const std::string& service, uint64_t flags) {
    if (initialized_) {
        cleanup(); // Clean up existing resources first
    }

    // Try different providers in priority order
    const char** providers = utils::getLibfabricProviders();
    int rc = -1;
    
    for (int i = 0; providers[i] != nullptr; i++) {
        std::cout << "[Libfabric] Trying provider: " << providers[i] << std::endl;
        
        // Initialize fabric
        struct fi_info* hints = fi_allocinfo();
        hints->ep_attr->type = FI_EP_MSG;
        hints->domain_attr->threading = FI_THREAD_SAFE;
        hints->fabric_attr->prov_name = strdup(providers[i]);
        
        rc = fi_getinfo(utils::getLibfabricVersion(), node.c_str(), service.c_str(), flags, hints, &info_);
        if (rc == 0) {
            std::cout << "[Libfabric] Successfully found provider: " << providers[i] << std::endl;
            fi_freeinfo(hints);
            break;
        } else {
            std::cout << "[Libfabric] Provider " << providers[i] << " failed: " << fi_strerror(rc) << std::endl;
            fi_freeinfo(hints);
        }
    }
    
    if (rc != 0) {
        std::cerr << "[Libfabric] No suitable provider found" << std::endl;
        return rc;
    }
    
    // Create fabric
    rc = fi_fabric(info_->fabric_attr, &fabric_, nullptr);
    if (rc != 0) {
        std::cerr << "[Libfabric] fi_fabric failed: " << fi_strerror(rc) << std::endl;
        fi_freeinfo(info_);
        info_ = nullptr;
        return rc;
    }
    
    // Create domain
    rc = fi_domain(fabric_, info_, &domain_, nullptr);
    if (rc != 0) {
        std::cerr << "[Libfabric] fi_domain failed: " << fi_strerror(rc) << std::endl;
        fi_close(&fabric_->fid);
        fabric_ = nullptr;
        fi_freeinfo(info_);
        info_ = nullptr;
        return rc;
    }
    
    // Create completion queue
    struct fi_cq_attr cq_attr = {0};
    cq_attr.size = utils::getDefaultCqSize();
    cq_attr.flags = FI_SELECTIVE_COMPLETION;
    rc = fi_cq_open(domain_, &cq_attr, &cq_, nullptr);
    if (rc != 0) {
        std::cerr << "[Libfabric] fi_cq_open failed: " << fi_strerror(rc) << std::endl;
        fi_close(&domain_->fid);
        domain_ = nullptr;
        fi_close(&fabric_->fid);
        fabric_ = nullptr;
        fi_freeinfo(info_);
        info_ = nullptr;
        return rc;
    }
    
    // Create event queue
    struct fi_eq_attr eq_attr = {0};
    eq_attr.size = utils::getDefaultCqSize();
    rc = fi_eq_open(fabric_, &eq_attr, &eq_, nullptr);
    if (rc != 0) {
        std::cerr << "[Libfabric] fi_eq_open failed: " << fi_strerror(rc) << std::endl;
        fi_close(&cq_->fid);
        cq_ = nullptr;
        fi_close(&domain_->fid);
        domain_ = nullptr;
        fi_close(&fabric_->fid);
        fabric_ = nullptr;
        fi_freeinfo(info_);
        info_ = nullptr;
        return rc;
    }
    
    initialized_ = true;
    return 0;
}

void LibfabricCommon::cleanup() {
    if (!initialized_) return;
    
    if (eq_) {
        fi_close(&eq_->fid);
        eq_ = nullptr;
    }
    if (cq_) {
        fi_close(&cq_->fid);
        cq_ = nullptr;
    }
    if (domain_) {
        fi_close(&domain_->fid);
        domain_ = nullptr;
    }
    if (fabric_) {
        fi_close(&fabric_->fid);
        fabric_ = nullptr;
    }
    if (info_) {
        fi_freeinfo(info_);
        info_ = nullptr;
    }
    
    initialized_ = false;
}

} // namespace hshm::lbm::libfabric 