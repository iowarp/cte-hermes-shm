#pragma once
#include "zmq_transport.h"
#include "thallium_transport.h"

namespace hshm::lbm {

inline std::unique_ptr<Client> TransportFactory::GetClient(const std::string &url, Transport t) {
    switch (t) {
        case Transport::kZeroMq:
            return std::make_unique<ZeroMqClient>(url);
        case Transport::kThallium:
            return std::make_unique<ThalliumClient>(url);
        default:
            return nullptr;
    }
}

inline std::unique_ptr<Server> TransportFactory::GetServer(const std::string &url, Transport t) {
    switch (t) {
        case Transport::kZeroMq:
            return std::make_unique<ZeroMqServer>(url);
        case Transport::kThallium:
            return std::make_unique<ThalliumServer>(url);
        default:
            return nullptr;
    }
}

} // namespace hshm::lbm 