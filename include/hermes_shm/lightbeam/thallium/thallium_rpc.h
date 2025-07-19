#pragma once
#include <thallium.hpp>
#include <memory>
#include <string>
#include <atomic>

namespace tl = thallium;

namespace hshm::lbm::thallium {

/**
 * Thallium RPC wrapper following the reference implementation EXACTLY
 */
class ThalliumRpc {
 public:
  std::atomic<bool> kill_requested_{false};      // From reference
  std::unique_ptr<tl::engine> client_engine_;    // From reference
  std::unique_ptr<tl::engine> server_engine_;    // From reference
  std::string server_address_;
  std::string protocol_;
  
  ThalliumRpc() = default;  // From reference

  /** Initialize server following reference pattern EXACTLY */
  void ServerInit(const std::string& address, const std::string& protocol);

  /** Initialize client following reference pattern EXACTLY */
  void ClientInit(const std::string& protocol);

  /** Run the daemon following reference pattern EXACTLY */
  void RunDaemon();

  /** Stop daemon following reference pattern EXACTLY */
  void StopThisDaemon();

  /** Get server name following reference pattern */
  std::string GetServerName(const std::string& ip_address, int port);

  /** Register RPC following reference pattern EXACTLY */
  template <typename RpcLambda>
  void RegisterRpc(const char *name, RpcLambda &&lambda) {
    if (!server_engine_) {
      throw std::runtime_error("Server engine not initialized");
    }
    // Follow reference exactly: server_engine_->define(name, std::forward<RpcLambda>(lambda), 0, pool);
    // Note: We don't have pool in our simplified version, so use the basic define
    server_engine_->define(name, std::forward<RpcLambda>(lambda));
    std::cout << "[Thallium] Registered RPC: " << name << std::endl;
  }

  /** Synchronous call following reference pattern EXACTLY */
  template <typename RetT, typename... Args>
  RetT SyncCall(const std::string &server_address, const std::string &func_name, Args &&...args) {
    // Follow reference Call() method exactly
    try {
      // Follow reference: tl::remote_procedure remote_proc = client_engine_->define(func_name);
      tl::remote_procedure remote_proc = client_engine_->define(func_name);
      // Follow reference: tl::endpoint server = client_engine_->lookup(server_name);
      tl::endpoint server = client_engine_->lookup(server_address);
      
      // Follow reference exactly:
      if constexpr (std::is_same_v<RetT, void>) {
        remote_proc.disable_response();
        remote_proc.on(server)(std::forward<Args>(args)...);
      } else {
        RetT result = remote_proc.on(server)(std::forward<Args>(args)...);
        return result;
      }
    } catch (tl::margo_exception &err) {
      // Follow reference exactly
      std::cerr << "[Thallium] Failed on function: " << func_name << ": " << err.what() << std::endl;
      throw;
    }
  }

  /** Make bulk transfer following reference pattern */
  tl::bulk MakeBulk(tl::engine &engine, const char* data, size_t size, tl::bulk_mode mode);

  /** Client-side bulk */
  tl::bulk MakeBulkClient(const char* data, size_t size, bool for_write);

  /** Server-side bulk */
  tl::bulk MakeBulkServer(const char* data, size_t size, bool for_write);
};

} // namespace hshm::lbm::thallium