#pragma once
#include "thallium_rpc.h"
#include "hermes_shm/lightbeam/types.h"
#include <memory>
#include <thread>
#include <atomic>
#include <cereal/types/string.hpp>

namespace hshm::lbm::thallium {

/**
 * Server implementation following your API requirements
 */
class Server {
 public:
  Server();
  ~Server();

  void StartServer(const std::string &url);
  void Stop();
  bool IsRunning() const { return running_; }

  template <typename Lambda>
  void RegisterRpc(const std::string &name, Lambda &&handler) {
    thallium_rpc_.RegisterRpc(name.c_str(), std::forward<Lambda>(handler));
  }
  
  

  // Make RunDaemon public so test can call it
  void RunDaemon();

 private:
  ThalliumRpc thallium_rpc_;
  std::atomic<bool> running_{false};
  std::thread daemon_thread_;
  
  std::string ParseAddressAndProtocol(const std::string &url, std::string &protocol);
};

} // namespace hshm::lbm::thallium