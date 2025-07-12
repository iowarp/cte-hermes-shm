#pragma once
#include "hermes_shm/lightbeam/base_server.h"
#include <memory>

namespace hshm::lbm::libfabric {

class Server : public hshm::lbm::IServer {
public:
  Server();
  ~Server();
  
  void StartServer(const std::string &url, TransportType transport = TransportType::AUTO) override;
  void Stop() override;
  void ProcessMessages() override;
  bool IsRunning() const override;
  
  // Additional methods for RDMA
  std::shared_ptr<MemoryRegion> RegisterMemory(void* addr, size_t length, uint64_t access_flags);
  void InitializeReceiveBuffers();
  void PostReceiveOperations();
  void ProcessCompletions();
  void SendEcho(const char* data, size_t size, uint64_t sequence);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace hshm::lbm::libfabric