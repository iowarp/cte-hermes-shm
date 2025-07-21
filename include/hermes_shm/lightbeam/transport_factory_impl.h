#pragma once
#include "zmq_transport.h"
#include "thallium_transport.h"

namespace hshm::lbm {

inline std::unique_ptr<Client> TransportFactory::GetClient(const std::string &addr, Transport t, const std::string &protocol, int port) {
    switch (t) {
        case Transport::kZeroMq:
            return std::make_unique<ZeroMqClient>(addr, protocol.empty() ? "tcp" : protocol, port == 0 ? 8192 : port);
        case Transport::kThallium:
            return std::make_unique<ThalliumClient>(addr, protocol.empty() ? "ofi+sockets" : protocol, port == 0 ? 8200 : port);
        default:
            return nullptr;
    }
}

inline std::unique_ptr<Client> TransportFactory::GetClient(const std::string &addr, Transport t, const std::string &protocol, int port, const std::string &domain) {
    switch (t) {
        case Transport::kZeroMq:
            return std::make_unique<ZeroMqClient>(addr, protocol.empty() ? "tcp" : protocol, port == 0 ? 8192 : port);
        case Transport::kThallium:
            return std::make_unique<ThalliumClient>(addr, protocol.empty() ? "ofi+sockets" : protocol, port == 0 ? 8200 : port, domain);
        default:
            return nullptr;
    }
}

inline std::unique_ptr<Server> TransportFactory::GetServer(const std::string &addr, Transport t, const std::string &protocol, int port) {
    switch (t) {
        case Transport::kZeroMq:
            return std::make_unique<ZeroMqServer>(addr, protocol.empty() ? "tcp" : protocol, port == 0 ? 8192 : port);
        case Transport::kThallium:
            return std::make_unique<ThalliumServer>(addr, protocol.empty() ? "ofi+sockets" : protocol, port == 0 ? 8200 : port);
        default:
            return nullptr;
    }
}

inline std::unique_ptr<Server> TransportFactory::GetServer(const std::string &addr, Transport t, const std::string &protocol, int port, const std::string &domain) {
    switch (t) {
        case Transport::kZeroMq:
            return std::make_unique<ZeroMqServer>(addr, protocol.empty() ? "tcp" : protocol, port == 0 ? 8192 : port);
        case Transport::kThallium:
            return std::make_unique<ThalliumServer>(addr, protocol.empty() ? "ofi+sockets" : protocol, port == 0 ? 8200 : port, domain);
        default:
            return nullptr;
    }
}

} // namespace hshm::lbm 