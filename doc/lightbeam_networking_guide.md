# HSHM Lightbeam Networking Guide

## Overview

Lightbeam is HSHM's high-performance networking abstraction layer that provides unified interfaces for different transport mechanisms including ZeroMQ, Thallium/Mercury, and Libfabric. It enables efficient data transfer across distributed systems with support for both traditional messaging and RDMA-style operations.

## Core Concepts

### Transport Types

Lightbeam supports multiple transport backends:

```cpp
#include "hermes_shm/lightbeam/lightbeam.h"

namespace hshm::lbm {
    enum class Transport { 
        kZeroMq,      // ZeroMQ messaging
        kThallium,    // Thallium/Mercury RPC
        kLibfabric    // Libfabric/OFI networking
    };
}
```

### Key Data Structures

```cpp
namespace hshm::lbm {
    
    // Event for asynchronous operations
    struct Event {
        bool is_done = false;
        int error_code = 0;
        std::string error_message;
        size_t bytes_transferred = 0;
    };
    
    // Memory bulk for data transfers
    struct Bulk {
        char* data;
        size_t size;
        int flags;
        void* desc = nullptr;    // RDMA memory registration descriptor
        void* mr = nullptr;      // Memory region handle (fid_mr*)
    };
}
```

## Basic Client-Server Communication

```cpp
#include "hermes_shm/lightbeam/lightbeam.h"
#include "hermes_shm/lightbeam/transport_factory_impl.h"

void basic_communication_example() {
    using namespace hshm::lbm;
    
    const std::string server_addr = "127.0.0.1";
    const std::string protocol = "tcp";
    const int port = 8888;
    
    // Create server
    auto server = TransportFactory::GetServer(server_addr, Transport::kZeroMq, 
                                             protocol, port);
    
    // Get actual server address (may include auto-assigned port)
    std::string actual_addr = server->GetAddress();
    printf("Server listening on: %s\n", actual_addr.c_str());
    
    // Create client
    auto client = TransportFactory::GetClient(actual_addr, Transport::kZeroMq, 
                                             protocol, port);
    
    // Prepare data to send
    const std::string message = "Hello from Lightbeam!";
    
    // Client: expose data and send
    Bulk send_bulk = client->Expose(message.data(), message.size(), 0);
    Event* send_event = client->Send(send_bulk);
    
    // Server: prepare receive buffer
    std::vector<char> receive_buffer(message.size());
    Bulk recv_bulk = server->Expose(receive_buffer.data(), receive_buffer.size(), 0);
    
    // Server: initiate receive
    Event* recv_event = server->Recv(recv_bulk);
    
    // Wait for completion
    while (!send_event->is_done || !recv_event->is_done) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    
    // Check results
    if (send_event->error_code == 0 && recv_event->error_code == 0) {
        std::string received_message(receive_buffer.data(), recv_event->bytes_transferred);
        printf("Successfully sent and received: '%s'\n", received_message.c_str());
    } else {
        printf("Error in communication\n");
    }
    
    // Cleanup
    delete send_event;
    delete recv_event;
}
```

## Best Practices

1. **Transport Selection**: Choose transport based on requirements:
   - **ZeroMQ**: Simple messaging, good for prototyping
   - **Thallium**: RPC-style communication, good for request/response
   - **Libfabric**: High-performance, RDMA-capable networking

2. **Error Handling**: Always check `Event::error_code` and implement retry logic for critical operations

3. **Memory Management**: 
   - Clean up `Event` objects after use
   - Be careful with `Bulk` lifetime - data must remain valid during transfer

4. **Performance**: 
   - Use connection pooling for frequently used endpoints
   - Implement batching for small messages
   - Consider async operations for high throughput

5. **Fault Tolerance**: Implement timeout and retry mechanisms for network operations

6. **Resource Management**: Monitor connection counts and clean up unused connections

7. **Protocol Selection**: Choose appropriate protocols:
   - `tcp/sockets`: General purpose, widely supported
   - `ofi+sockets`: OFI over sockets, good fallback
   - `verbs/mlx`: InfiniBand for high-performance clusters

8. **Data Integrity**: Implement checksums or validation for critical data transfers

9. **Scalability**: Design for distributed deployment with service discovery

10. **Monitoring**: Implement comprehensive logging and metrics collection for network operations