// server.h - Updated to include Recv method
#pragma once

#include "types.h"
#include <string>
#include <memory>

namespace hshm::lbm {

struct Server {
    Server();
    ~Server();
    
    // Initialize the server and allow clients to connect and send data to us.
    // This should start a daemonized process.
    // url: provider://domain:address:port. For example, tcp://lo:127.0.0.1:8192
    void StartServer(const std::string &url);
    
    // Stop the server
    void Stop();
    
    // Check if server is running
    bool IsRunning() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace hshm::lbm