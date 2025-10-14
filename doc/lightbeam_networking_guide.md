# HSHM Lightbeam Networking Guide

## Overview

Lightbeam is HSHM's high-performance networking abstraction layer that provides a unified interface for distributed data transfer. The current implementation supports ZeroMQ as the transport mechanism, with a two-phase messaging protocol that separates metadata from bulk data transfers.

## Core Concepts

### Two-Phase Messaging Protocol

Lightbeam uses a two-phase approach to message transmission:

1. **Metadata Phase**: Sends message metadata including bulk descriptors
2. **Bulk Data Phase**: Transfers the actual data payloads

This separation allows receivers to:
- Inspect message metadata before allocating buffers
- Allocate appropriately sized buffers based on incoming data sizes
- Handle multiple data chunks efficiently

### Transport Types

Currently supported transport:

```cpp
#include <hermes_shm/lightbeam/zmq_transport.h>

namespace hshm::lbm {
    enum class Transport {
        kZeroMq      // ZeroMQ messaging
    };
}
```

## Data Structures

### Event

Represents the status of an asynchronous operation:

```cpp
struct Event {
    bool is_done = false;              // Operation completion flag
    int error_code = 0;                // 0 for success, non-zero for errors
    std::string error_message;         // Human-readable error description
    size_t bytes_transferred = 0;      // Total bytes transferred
};
```

### Bulk

Describes a memory region for data transfer:

```cpp
struct Bulk {
    hipc::FullPtr<char> data;    // Pointer to data (supports shared memory)
    size_t size;                 // Size of data in bytes
    int flags;                   // Transfer flags
    void* desc = nullptr;        // RDMA memory registration descriptor
    void* mr = nullptr;          // Memory region handle (for future RDMA support)
};
```

**Key Features:**
- Uses `hipc::FullPtr` for shared memory compatibility
- Can be created from raw pointers, `hipc::Pointer`, or `hipc::FullPtr`
- Prepared for future RDMA transport extensions

### LbmMeta

Base class for message metadata:

```cpp
class LbmMeta {
 public:
    std::vector<Bulk> bulks;  // Collection of bulk data descriptors
};
```

**Usage:**
- Extend `LbmMeta` to include custom metadata fields
- Must implement cereal serialization for custom fields
- The `bulks` vector describes all data payloads in the message

## API Reference

### Client Interface

The client initiates data transfers:

```cpp
class Client {
 public:
    // Expose memory for transfer (creates Bulk descriptor)
    virtual Bulk Expose(const char* data, size_t data_size, int flags) = 0;
    virtual Bulk Expose(const hipc::Pointer& ptr, size_t data_size, int flags) = 0;
    virtual Bulk Expose(const hipc::FullPtr<char>& ptr, size_t data_size, int flags) = 0;

    // Send metadata and bulk data
    template<typename MetaT>
    Event* Send(MetaT &meta);
};
```

**Methods:**
- `Expose()`: Registers memory for transfer, returns `Bulk` descriptor
  - Accepts raw pointers, `hipc::Pointer`, or `hipc::FullPtr`
  - Returns immediately (no actual data transfer)
- `Send()`: Transmits metadata and all associated bulks
  - Template method accepting any `LbmMeta`-derived type
  - Serializes metadata using cereal
  - Returns `Event*` for tracking operation status

### Server Interface

The server receives data transfers:

```cpp
class Server {
 public:
    // Expose memory for receiving data
    virtual Bulk Expose(char* data, size_t data_size, int flags) = 0;
    virtual Bulk Expose(const hipc::Pointer& ptr, size_t data_size, int flags) = 0;
    virtual Bulk Expose(const hipc::FullPtr<char>& ptr, size_t data_size, int flags) = 0;

    // Two-phase receive
    template<typename MetaT>
    Event* RecvMetadata(MetaT &meta);

    template<typename MetaT>
    Event* RecvBulks(MetaT &meta);

    // Get server address
    virtual std::string GetAddress() const = 0;
};
```

**Methods:**
- `Expose()`: Registers receive buffers, returns `Bulk` descriptor
- `RecvMetadata()`: Receives and deserializes message metadata
  - Returns immediately with `is_done=false` if no message available
  - Populates `meta.bulks` with sender's bulk descriptors
- `RecvBulks()`: Receives actual data into exposed buffers
  - Must be called after `RecvMetadata()` completes
  - Requires buffers to be exposed with matching sizes
  - Returns immediately if no data available
- `GetAddress()`: Returns the server's bind address

### TransportFactory

Factory for creating client and server instances:

```cpp
class TransportFactory {
 public:
    static std::unique_ptr<Client> GetClient(
        const std::string& addr, Transport t,
        const std::string& protocol = "", int port = 0);

    static std::unique_ptr<Server> GetServer(
        const std::string& addr, Transport t,
        const std::string& protocol = "", int port = 0);
};
```

## Examples

### Basic Client-Server Communication

```cpp
#include <hermes_shm/lightbeam/zmq_transport.h>
#include <iostream>
#include <vector>
#include <thread>
#include <chrono>

using namespace hshm::lbm;

void basic_example() {
    // Server setup
    std::string addr = "127.0.0.1";
    std::string protocol = "tcp";
    int port = 8888;

    auto server = TransportFactory::GetServer(addr, Transport::kZeroMq,
                                             protocol, port);
    auto client = TransportFactory::GetClient(addr, Transport::kZeroMq,
                                             protocol, port);

    // Give ZMQ time to establish connection
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // CLIENT: Prepare and send data
    const char* message = "Hello, Lightbeam!";
    size_t message_size = strlen(message);

    LbmMeta send_meta;
    Bulk bulk = client->Expose(message, message_size, 0);
    send_meta.bulks.push_back(bulk);

    Event* send_event = client->Send(send_meta);
    assert(send_event->is_done);
    assert(send_event->error_code == 0);
    std::cout << "Client sent " << send_event->bytes_transferred << " bytes\n";
    delete send_event;

    // SERVER: Receive metadata
    LbmMeta recv_meta;
    Event* recv_event = nullptr;
    while (!recv_event || !recv_event->is_done) {
        if (recv_event) delete recv_event;
        recv_event = server->RecvMetadata(recv_meta);
        if (!recv_event->is_done) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    assert(recv_event->error_code == 0);
    std::cout << "Server received metadata with "
              << recv_meta.bulks.size() << " bulks\n";
    delete recv_event;

    // SERVER: Allocate buffer and receive data
    std::vector<char> buffer(recv_meta.bulks[0].size);
    recv_meta.bulks[0] = server->Expose(buffer.data(), buffer.size(), 0);

    recv_event = server->RecvBulks(recv_meta);
    while (!recv_event->is_done) {
        delete recv_event;
        recv_event = server->RecvBulks(recv_meta);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    assert(recv_event->error_code == 0);
    std::cout << "Server received: "
              << std::string(buffer.data(), buffer.size()) << "\n";
    delete recv_event;
}
```

### Custom Metadata with Multiple Bulks

```cpp
#include <hermes_shm/lightbeam/zmq_transport.h>
#include <cereal/archives/binary.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/vector.hpp>

using namespace hshm::lbm;

// Custom metadata class
class RequestMeta : public LbmMeta {
 public:
    int request_id;
    std::string operation;
    std::string client_name;
};

// Cereal serialization
namespace cereal {
    template<class Archive>
    void serialize(Archive& ar, RequestMeta& meta) {
        ar(meta.bulks, meta.request_id, meta.operation, meta.client_name);
    }
}

void custom_metadata_example() {
    auto server = std::make_unique<ZeroMqServer>("127.0.0.1", "tcp", 8889);
    auto client = std::make_unique<ZeroMqClient>("127.0.0.1", "tcp", 8889);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // CLIENT: Send multiple data chunks with metadata
    const char* data1 = "First chunk";
    const char* data2 = "Second chunk";

    RequestMeta send_meta;
    send_meta.request_id = 42;
    send_meta.operation = "write";
    send_meta.client_name = "client_01";

    send_meta.bulks.push_back(client->Expose(data1, strlen(data1), 0));
    send_meta.bulks.push_back(client->Expose(data2, strlen(data2), 0));

    Event* send_event = client->Send(send_meta);
    assert(send_event->is_done);
    delete send_event;

    // SERVER: Receive metadata
    RequestMeta recv_meta;
    Event* recv_event = nullptr;
    while (!recv_event || !recv_event->is_done) {
        if (recv_event) delete recv_event;
        recv_event = server->RecvMetadata(recv_meta);
    }

    std::cout << "Request ID: " << recv_meta.request_id << "\n";
    std::cout << "Operation: " << recv_meta.operation << "\n";
    std::cout << "Client: " << recv_meta.client_name << "\n";
    std::cout << "Number of bulks: " << recv_meta.bulks.size() << "\n";
    delete recv_event;

    // SERVER: Allocate buffers and receive bulks
    std::vector<std::vector<char>> buffers;
    for (size_t i = 0; i < recv_meta.bulks.size(); ++i) {
        buffers.emplace_back(recv_meta.bulks[i].size);
        recv_meta.bulks[i] = server->Expose(buffers[i].data(),
                                            buffers[i].size(), 0);
    }

    recv_event = server->RecvBulks(recv_meta);
    while (!recv_event->is_done) {
        delete recv_event;
        recv_event = server->RecvBulks(recv_meta);
    }

    for (size_t i = 0; i < buffers.size(); ++i) {
        std::cout << "Chunk " << i << ": "
                  << std::string(buffers[i].begin(), buffers[i].end()) << "\n";
    }
    delete recv_event;
}
```

### Working with Shared Memory Pointers

```cpp
#include <hermes_shm/lightbeam/zmq_transport.h>
#include <hermes_shm/memory/memory_manager.h>

using namespace hshm::lbm;

void shared_memory_example() {
    // Assume memory manager is initialized
    hipc::Allocator* alloc = HSHM_MEMORY_MANAGER->GetDefaultAllocator();

    // Allocate shared memory
    size_t data_size = 1024;
    hipc::Pointer shm_ptr = alloc->Allocate(data_size);
    hipc::FullPtr<char> full_ptr(shm_ptr);

    // Write data to shared memory
    memcpy(full_ptr.ptr_, "Shared memory data", 18);

    // Create client and expose shared memory
    auto client = std::make_unique<ZeroMqClient>("127.0.0.1", "tcp", 8890);

    LbmMeta meta;
    // Can use either hipc::Pointer or hipc::FullPtr directly
    meta.bulks.push_back(client->Expose(full_ptr, data_size, 0));

    Event* event = client->Send(meta);
    assert(event->is_done);
    delete event;

    // Free shared memory
    alloc->Free(shm_ptr);
}
```

### Distributed MPI Communication

```cpp
#include <hermes_shm/lightbeam/zmq_transport.h>
#include <mpi.h>

using namespace hshm::lbm;

void distributed_example() {
    MPI_Init(nullptr, nullptr);

    int my_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    std::string addr = "127.0.0.1";
    int base_port = 9000;

    // Each rank creates a server on a unique port
    auto server = TransportFactory::GetServer(
        addr, Transport::kZeroMq, "tcp", base_port + my_rank);

    // Rank 0 sends to all other ranks
    if (my_rank == 0) {
        std::vector<std::unique_ptr<Client>> clients;
        for (int i = 1; i < world_size; ++i) {
            clients.push_back(TransportFactory::GetClient(
                addr, Transport::kZeroMq, "tcp", base_port + i));
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(200));

        for (size_t i = 0; i < clients.size(); ++i) {
            std::string msg = "Message to rank " + std::to_string(i + 1);

            LbmMeta meta;
            meta.bulks.push_back(clients[i]->Expose(msg.data(), msg.size(), 0));

            Event* event = clients[i]->Send(meta);
            assert(event->is_done);
            delete event;
        }
    } else {
        // Other ranks receive from rank 0
        LbmMeta meta;
        Event* event = nullptr;

        while (!event || !event->is_done) {
            if (event) delete event;
            event = server->RecvMetadata(meta);
        }

        std::vector<char> buffer(meta.bulks[0].size);
        meta.bulks[0] = server->Expose(buffer.data(), buffer.size(), 0);

        delete event;
        event = server->RecvBulks(meta);
        while (!event->is_done) {
            delete event;
            event = server->RecvBulks(meta);
        }

        std::cout << "Rank " << my_rank << " received: "
                  << std::string(buffer.begin(), buffer.end()) << "\n";
        delete event;
    }

    MPI_Finalize();
}
```

## Best Practices

### 1. Connection Management

```cpp
// Give ZMQ time to establish connections
std::this_thread::sleep_for(std::chrono::milliseconds(100));

// Store clients/servers in containers for reuse
std::vector<std::unique_ptr<Client>> client_pool;
```

### 2. Error Handling

```cpp
Event* event = client->Send(meta);
if (event->error_code != 0) {
    std::cerr << "Send failed: " << event->error_message << "\n";
    // Implement retry logic
}
assert(event->is_done);  // For synchronous operations
delete event;
```

### 3. Non-Blocking Operations

```cpp
// Poll for completion instead of blocking
Event* recv_event = nullptr;
while (!recv_event || !recv_event->is_done) {
    if (recv_event) delete recv_event;
    recv_event = server->RecvMetadata(meta);
    if (!recv_event->is_done) {
        // Do other work or sleep briefly
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}
```

### 4. Memory Management

```cpp
// Always delete Event objects
Event* event = client->Send(meta);
// ... use event ...
delete event;

// Ensure data lifetime during transfer
{
    std::vector<char> data(1024);
    Bulk bulk = client->Expose(data.data(), data.size(), 0);
    // data must remain valid until Send() completes
    Event* event = client->Send(meta);
    delete event;
} // data destroyed after Send completes
```

### 5. Custom Metadata Serialization

```cpp
// Always serialize bulks first in custom metadata
namespace cereal {
    template<class Archive>
    void serialize(Archive& ar, CustomMeta& meta) {
        ar(meta.bulks);  // Serialize base class bulks first
        ar(meta.custom_field1, meta.custom_field2);  // Then custom fields
    }
}
```

### 6. Buffer Allocation Strategy

```cpp
// Allocate receive buffers based on metadata
LbmMeta meta;
Event* event = server->RecvMetadata(meta);
// ... wait for completion ...

// Now we know exact sizes needed
std::vector<std::vector<char>> buffers;
for (const auto& bulk : meta.bulks) {
    buffers.emplace_back(bulk.size);  // Allocate exact size
}
```

### 7. Multi-Threading

```cpp
// Use separate server thread for receiving
std::thread server_thread([&server]() {
    while (running) {
        LbmMeta meta;
        Event* event = server->RecvMetadata(meta);
        if (event->is_done && event->error_code == 0) {
            // Process message
        }
        delete event;
    }
});
```

## Performance Considerations

1. **Metadata Overhead**: Keep custom metadata small - it's serialized/deserialized on every message

2. **Bulk Count**: Minimize the number of bulks per message when possible

3. **Buffer Reuse**: Reuse allocated buffers across multiple receives

4. **Connection Pooling**: Create clients once and reuse them

5. **Serialization Cost**: Use efficient serialization for custom metadata

6. **ZMQ Socket Options**: Configure ZMQ for your use case (high water marks, linger, etc.)

## Limitations and Future Work

**Current Limitations:**
- Only ZeroMQ transport is implemented
- Synchronous operations only (non-blocking via polling)
- No built-in timeout mechanism
- Limited to TCP protocol

**Future Enhancements:**
- Thallium/Mercury transport for RPC-style communication
- Libfabric transport for RDMA operations
- Async/await style API
- Built-in timeout and retry mechanisms
- Protocol negotiation and versioning
- Connection multiplexing
