#pragma once
#include "thallium_rpc.h"
#include "hermes_shm/lightbeam/types.h"
#include <memory>
#include <vector>
#include <unordered_map>
#include <cereal/types/string.hpp>

namespace hshm::lbm::thallium {

/**
 * Client implementation following your API requirements
 */
class Client {
 public:
  Client();
  ~Client();

  void Connect(const std::string &url);
  void Disconnect(const std::string &url);
  Bulk Expose(const std::string &url, const char *data, size_t data_size, int flags);
  std::unique_ptr<Event> Send(const Bulk &bulk);
  std::unique_ptr<Event> Recv(char *buffer, size_t buffer_size, const std::string &from_url);
  

  // Synchronous RPC calls
  template <typename RetT, typename... Args>
  RetT SyncCall(const std::string &url, const std::string &rpc_name, Args&&... args) {
    return thallium_rpc_.SyncCall<RetT>(url, rpc_name, std::forward<Args>(args)...);
  }

 private:
  ThalliumRpc thallium_rpc_;
  std::vector<std::unique_ptr<Event>> active_events_;
  uint64_t next_event_id_ = 1;
  
  struct ThalliumBulkData {
    std::unique_ptr<tl::bulk> bulk;
  };
  std::unordered_map<uint64_t, std::unique_ptr<ThalliumBulkData>> bulk_data_;
};

} // namespace hshm::lbm::thallium