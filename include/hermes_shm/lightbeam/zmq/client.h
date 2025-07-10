#pragma once
#include "hermes_shm/lightbeam/base_client.h"
#include <memory>

namespace hshm::lbm::zmq {

class Client : public hshm::lbm::IClient {
 public:
  Client();
  void Connect(const std::string &url, hshm::lbm::TransportType transport) override;
  void Disconnect(const std::string &url) override;
  hshm::lbm::Bulk Expose(const std::string &url, const char *data, size_t data_size, int flags) override;
  hshm::lbm::Event* Send(const hshm::lbm::Bulk &bulk) override;
  hshm::lbm::Event* Recv(char *buffer, size_t buffer_size, const std::string &from_url) override;
  void ProcessCompletions() override;
  ~Client() override;
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace hshm::lbm::zmq 