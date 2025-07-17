#pragma once
#include "types.h"
#include <string>
#include <memory>

namespace hshm::lbm {

class Client {
public:
    Client();
    ~Client();
    
    // Connect to a server for bidirectional communication
    // The connection state is saved in an unordered map in this class.
    void Connect(const std::string &url);
    
    // Disconnect from a server
    void Disconnect(const std::string &url);
    
    // Exposes a data segment to the server at url
    // flags: A flag should exist to enable / disable RDMA.
    Bulk Expose(const std::string &url, const char *data, size_t data_size, int flags);
    
    // Send the bulk to the server. This should be non-blocking. This does not spawn threads. 
    // It should store an Event* internally in this object and return it to the user. 
    // The user can then poll the Event* to see when a specific I/O op completes and its error information.
    Event* Send(const Bulk &bulk);
    
    // Receive a bulk from the server. This should be non-blocking. This does not spawn threads. 
    // It should store an Event* internally and return it to the user. 
    // The user can then poll the Event* to see when a specific I/O op completes and its error information.
    Event* Recv(char *buffer, size_t buffer_size, const std::string &from_url);
    
    // When either a send or receive completes, it should update some internal data structure
    // This function will poll that data structure to see the status of different sends and receives
    // It will fill the Event* data structures with any error information
    void ProcessCompletions();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace hshm::lbm 