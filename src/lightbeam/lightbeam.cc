#include "hermes_shm/lightbeam/lightbeam.h"
#include <zmq.h>
#include <cassert>
#include <iostream>

namespace hshm::lbm {

struct Server::Impl {
  void *zmq_ctx = nullptr;
  void *zmq_sock = nullptr;
  bool running = false;
};

struct Client::Impl {
  void *zmq_ctx = nullptr;
  void *zmq_sock = nullptr;
};

Server::Server() : impl_(std::make_unique<Impl>()) {}

void Server::StartServer(const std::string &url, TransportType transport) {
  if (!impl_) impl_ = std::make_unique<Impl>();
  assert(!impl_->running);
  impl_->zmq_ctx = zmq_ctx_new();
  impl_->zmq_sock = zmq_socket(impl_->zmq_ctx, ZMQ_REP);
  int rc = zmq_bind(impl_->zmq_sock, url.c_str());
  if (rc != 0) {
    std::cerr << "ZeroMQ bind failed: " << zmq_strerror(zmq_errno()) << std::endl;
    return;
  }
  impl_->running = true;
}

void Server::Stop() {
  if (!impl_ || !impl_->running) return;
  zmq_close(impl_->zmq_sock);
  zmq_ctx_term(impl_->zmq_ctx);
  impl_->running = false;
}

void Server::ProcessMessages() {
  if (!impl_ || !impl_->running) return;
  zmq_pollitem_t items[] = {{impl_->zmq_sock, 0, ZMQ_POLLIN, 0}};
  int rc = zmq_poll(items, 1, 0); // Non-blocking poll
  if (rc > 0 && (items[0].revents & ZMQ_POLLIN)) {
    char buffer[1024];
    int n = zmq_recv(impl_->zmq_sock, buffer, sizeof(buffer), ZMQ_DONTWAIT);
    if (n > 0) {
      // Echo back for now
      zmq_send(impl_->zmq_sock, buffer, n, ZMQ_DONTWAIT);
    }
  }
}

bool Server::IsRunning() const {
  return impl_ && impl_->running;
}

Server::~Server() {
  if (impl_ && impl_->running) {
    Stop();
  }
}

Client::Client() : impl_(std::make_unique<Impl>()) {}

void Client::Connect(const std::string &url, TransportType transport) {
  if (!impl_) impl_ = std::make_unique<Impl>();
  impl_->zmq_ctx = zmq_ctx_new();
  impl_->zmq_sock = zmq_socket(impl_->zmq_ctx, ZMQ_REQ);
  int rc = zmq_connect(impl_->zmq_sock, url.c_str());
  if (rc != 0) {
    std::cerr << "ZeroMQ connect failed: " << zmq_strerror(zmq_errno()) << std::endl;
    return;
  }
}

void Client::Disconnect(const std::string &url) {
  if (!impl_) return;
  if (impl_->zmq_sock) {
    zmq_close(impl_->zmq_sock);
    impl_->zmq_sock = nullptr;
  }
  if (impl_->zmq_ctx) {
    zmq_ctx_term(impl_->zmq_ctx);
    impl_->zmq_ctx = nullptr;
  }
}

Bulk Client::Expose(const std::string &url, const char *data, size_t data_size, int flags) {
  Bulk bulk;
  bulk.data = const_cast<char*>(data);
  bulk.size = data_size;
  bulk.target_url = url;
  bulk.preferred_transport = TransportType::TCP;
  bulk.zmq_handle = impl_ ? impl_->zmq_sock : nullptr;
  return bulk;
}

Event* Client::Send(const Bulk &bulk) {
  if (!impl_) return nullptr;
  int rc = zmq_send(impl_->zmq_sock, bulk.data, bulk.size, ZMQ_DONTWAIT);
  static Event event;
  event.is_done = (rc == (int)bulk.size);
  event.error_code = (rc < 0) ? zmq_errno() : 0;
  event.error_message = (rc < 0) ? zmq_strerror(zmq_errno()) : "";
  event.bytes_transferred = (rc > 0) ? rc : 0;
  event.transport_used = TransportType::TCP;
  return &event;
}

Event* Client::Recv(char *buffer, size_t buffer_size, const std::string &from_url) {
  if (!impl_) return nullptr;
  int rc = zmq_recv(impl_->zmq_sock, buffer, buffer_size, ZMQ_DONTWAIT);
  static Event event;
  event.is_done = (rc > 0);
  event.error_code = (rc < 0) ? zmq_errno() : 0;
  event.error_message = (rc < 0) ? zmq_strerror(zmq_errno()) : "";
  event.bytes_transferred = (rc > 0) ? rc : 0;
  event.transport_used = TransportType::TCP;
  return &event;
}

void Client::ProcessCompletions() {
  // For ZeroMQ, completions are handled by polling in Send/Recv
}

Client::~Client() {
  if (impl_) {
    if (impl_->zmq_sock) {
      zmq_close(impl_->zmq_sock);
      impl_->zmq_sock = nullptr;
    }
    if (impl_->zmq_ctx) {
      zmq_ctx_term(impl_->zmq_ctx);
      impl_->zmq_ctx = nullptr;
    }
  }
}

} // namespace hshm::lbm