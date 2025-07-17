// base_server.h - Updated interface to include Recv function
#pragma once

#include "types.h"
#include <string>

namespace hshm::lbm {

class IServer {
public:
    virtual ~IServer() = default;
    
    // Initialize the server and allow clients to connect and send data to us.
    // This should start a daemonized process.
    // url: provider://domain:address:port. For example, tcp://lo:127.0.0.1:8192
    virtual void StartServer(const std::string &url, TransportType transport) = 0;
    
    // Stop the server
    virtual void Stop() = 0;
    
    // Process incoming messages (original function - for compatibility)
    virtual void ProcessMessages() = 0;
    
    // Check if server is running
    virtual bool IsRunning() const = 0;
    
    // NEW: Receive an event on the server
    // Non-blocking
    // Return true if recv did something
    virtual bool Recv(const Bulk &bulk) = 0;
};

} // namespace hshm::lbm