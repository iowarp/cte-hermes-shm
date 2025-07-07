#pragma once
#include <string>
#include <memory>

namespace hshm::lbm {

/** \brief Supported transport types for LightBeam. */
enum class TransportType { TCP, RDMA, AUTO };

/** \brief Bulk transfer flags. */
enum BulkFlags {
    LBM_RDMA_ENABLE = 1,
    LBM_RDMA_DISABLE = 2,
    LBM_ZERO_COPY = 4
};

/** \brief Event structure for async operations. */
struct Event {
    bool is_done = false;
    int error_code = 0;
    std::string error_message;
    size_t bytes_transferred = 0;
    TransportType transport_used = TransportType::AUTO;
};

/** \brief Bulk transfer descriptor. */
struct Bulk {
    char *data = nullptr;
    size_t size = 0;
    std::string target_url;
    TransportType preferred_transport = TransportType::AUTO;
    void *zmq_handle = nullptr; // ZeroMQ specific
};

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