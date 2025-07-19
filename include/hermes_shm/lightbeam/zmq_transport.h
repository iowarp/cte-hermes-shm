#pragma once
#include <zmq.h>
#include "lightbeam.h"
#include <queue>
#include <mutex>
#include <memory>

namespace hshm::lbm {

class ZeroMqClient : public Client {
public:
    explicit ZeroMqClient(const std::string &url)
        : url_(url), ctx_(zmq_ctx_new()), socket_(zmq_socket(ctx_, ZMQ_PUSH)) {
        zmq_connect(socket_, url.c_str());
    }
    ~ZeroMqClient() override {
        zmq_close(socket_);
        zmq_ctx_destroy(ctx_);
    }
    Bulk Expose(const char *data, size_t data_size, int flags) override {
        Bulk bulk;
        bulk.data = const_cast<char*>(data);
        bulk.size = data_size;
        bulk.flags = flags;
        return bulk;
    }
    Event* Send(const Bulk &bulk) override {
        auto event = std::make_unique<Event>();
        int rc = zmq_send(socket_, bulk.data, bulk.size, ZMQ_DONTWAIT);
        if (rc == -1) {
            event->is_done = true;
            event->error_code = zmq_errno();
            event->error_message = zmq_strerror(event->error_code);
        } else {
            event->is_done = true;
            event->bytes_transferred = rc;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        events_.push(std::move(event));
        return events_.back().get();
    }

private:
    std::string url_;
    void *ctx_;
    void *socket_;
    std::queue<std::unique_ptr<Event>> events_;
    std::mutex mutex_;
};

class ZeroMqServer : public Server {
public:
    explicit ZeroMqServer(const std::string &url)
        : url_(url), ctx_(zmq_ctx_new()), socket_(zmq_socket(ctx_, ZMQ_PULL)) {
        zmq_bind(socket_, url.c_str());
    }
    ~ZeroMqServer() override {
        zmq_close(socket_);
        zmq_ctx_destroy(ctx_);
    }
    Bulk Expose(char *data, size_t data_size, int flags) override {
        Bulk bulk;
        bulk.data = data;
        bulk.size = data_size;
        bulk.flags = flags;
        return bulk;
    }
    Event* Recv(const Bulk &bulk) override {
        auto event = std::make_unique<Event>();
        int rc = zmq_recv(socket_, bulk.data, bulk.size, ZMQ_DONTWAIT);
        if (rc > 0) {
            event->is_done = true;
            event->bytes_transferred = rc;
        } else if (rc == -1 && zmq_errno() == EAGAIN) {
            // No data available, not an error
            event->is_done = false;
        } else {
            event->is_done = true;
            event->error_code = zmq_errno();
            event->error_message = zmq_strerror(event->error_code);
        }
        std::lock_guard<std::mutex> lock(mutex_);
        events_.push(std::move(event));
        return events_.back().get();
    }
    
    std::string GetAddress() const override {
        return url_;
    }
private:
    std::string url_;
    void *ctx_;
    void *socket_;
    std::queue<std::unique_ptr<Event>> events_;
    std::mutex mutex_;
};

// Factory implementation for ZMQ
inline std::unique_ptr<Client> TransportFactory::GetClient(const std::string &url, Transport t) {
    if (t == Transport::kZeroMq) {
        return std::make_unique<ZeroMqClient>(url);
    }
    return nullptr;
}
inline std::unique_ptr<Server> TransportFactory::GetServer(const std::string &url, Transport t) {
    if (t == Transport::kZeroMq) {
        return std::make_unique<ZeroMqServer>(url);
    }
    return nullptr;
}

} // namespace hshm::lbm 