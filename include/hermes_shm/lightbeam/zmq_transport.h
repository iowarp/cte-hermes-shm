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
        Event* event = new Event();
        int rc = zmq_send(socket_, bulk.data, bulk.size, ZMQ_DONTWAIT);
        if (rc == -1) {
            event->is_done = true;
            event->error_code = zmq_errno();
            event->error_message = zmq_strerror(event->error_code);
        } else {
            event->is_done = true;
            event->bytes_transferred = rc;
        }
        return event;
    }

private:
    std::string url_;
    void *ctx_;
    void *socket_;
};

class ZeroMqServer : public Server {
public:
    explicit ZeroMqServer(const std::string &url)
        : url_(url), ctx_(zmq_ctx_new()), socket_(zmq_socket(ctx_, ZMQ_PULL)) {
        int rc = zmq_bind(socket_, url.c_str());
        if (rc == -1) {
            std::string err = "ZeroMqServer failed to bind to URL '" + url + "': " + zmq_strerror(zmq_errno());
            zmq_close(socket_);
            zmq_ctx_destroy(ctx_);
            throw std::runtime_error(err);
        }
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
        Event* event = new Event();
        int rc = zmq_recv(socket_, bulk.data, bulk.size, ZMQ_DONTWAIT);
        if (rc > 0) {
            event->is_done = true;
            event->bytes_transferred = rc;
        } else if (rc == -1 && zmq_errno() == EAGAIN) {
            event->is_done = false;
        } else {
            event->is_done = true;
            event->error_code = zmq_errno();
            event->error_message = zmq_strerror(event->error_code);
        }
        return event;
    }
    
    std::string GetAddress() const override {
        return url_;
    }
private:
    std::string url_;
    void *ctx_;
    void *socket_;
};

} // namespace hshm::lbm 