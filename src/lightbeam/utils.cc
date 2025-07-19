#include "hermes_shm/lightbeam/utils.h"
#include <rdma/fabric.h>

namespace hshm::lbm::utils {

std::pair<std::string, std::string> parseUrl(const std::string& url) {
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

const char** getLibfabricProviders() {
    // Static array of providers in priority order (sockets is most reliable for testing)
    static const char* providers[] = {"sockets", "tcp", "udp", nullptr};
    return providers;
}

uint32_t getLibfabricVersion() {
    return FI_VERSION(1, 1);
}

} // namespace hshm::lbm::utils 