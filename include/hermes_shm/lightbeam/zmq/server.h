#pragma once
#include "hermes_shm/lightbeam/base_server.h"
#include <memory>

namespace hshm::lbm::zmq {

class Server : public hshm::lbm::IServer {
 public:
  Server();
  void StartServer(const std::string &url, hshm::lbm::TransportType transport) override;
  void Stop() override;
  void ProcessMessages() override;
  bool IsRunning() const override;
  ~Server() override;
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace hshm::lbm::zmq 