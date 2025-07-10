#pragma once
#include <memory>
#include <string>
#include "hermes_shm/lightbeam/types.h"

namespace hshm::lbm {

class Client {
public:
    Client();
    void Connect(const std::string &url, TransportType transport = TransportType::AUTO);
    void Disconnect(const std::string &url);
    Bulk Expose(const std::string &url, const char *data, size_t data_size, int flags);
    Event* Send(const Bulk &bulk);
    Event* Recv(char *buffer, size_t buffer_size, const std::string &from_url);
    void ProcessCompletions();
    ~Client();
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace hshm::lbm 