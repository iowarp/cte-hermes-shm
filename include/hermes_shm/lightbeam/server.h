#pragma once
#include <memory>
#include <string>
#include "hermes_shm/lightbeam/types.h"

namespace hshm::lbm {

struct Server {
    Server();
    void StartServer(const std::string &url, TransportType transport = TransportType::AUTO);
    void Stop();
    void ProcessMessages();
    bool IsRunning() const;
    ~Server();
 private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace hshm::lbm 