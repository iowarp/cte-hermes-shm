#pragma once
#include "hermes_shm/lightbeam/base_client.h"
#include <memory>

namespace hshm::lbm::zmq {

class Client : public hshm::lbm::IClient {
public:
  Client();
  ~Client();
  
  void Connect(const std::string &url, TransportType transport = TransportType::AUTO) override;
  void Disconnect(const std::string &url) override;
  Bulk Expose(const std::string &url, const char *data, size_t data_size, int flags) override;
  std::unique_ptr<Event> Send(const Bulk &bulk) override;
  std::unique_ptr<Event> Recv(char *buffer, size_t buffer_size, const std::string &from_url) override;
  void ProcessCompletions(double timeout_msec = 0.0) override;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace hshm::lbm::zmq