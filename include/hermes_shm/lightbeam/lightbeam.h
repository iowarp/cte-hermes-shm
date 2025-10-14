#pragma once
// Common types, interfaces, and factory for lightbeam transports.
// Users must include the appropriate transport header (zmq_transport.h)
// before using the factory for that transport.
#include <string>
#include <memory>
#include <vector>
#include <cstring>
#include <cassert>
#include "hermes_shm/memory/memory_manager.h"

namespace hshm::lbm {

// --- Types ---
struct Event {
  bool is_done = false;
  int error_code = 0;
  std::string error_message;
  size_t bytes_transferred = 0;
};

struct Bulk {
  hipc::FullPtr<char> data;
  size_t size;
  int flags;
  void* desc = nullptr;  // For RDMA memory registration
  void* mr = nullptr;    // For RDMA memory region handle (fid_mr*)
};

// --- Metadata Base Class ---
class LbmMeta {
 public:
  std::vector<Bulk> bulks;
};

// --- Interfaces ---
class Client {
 public:
  virtual ~Client() = default;

  // Expose from raw pointer
  virtual Bulk Expose(const char* data, size_t data_size, int flags) = 0;

  // Expose from hipc::Pointer
  virtual Bulk Expose(const hipc::Pointer& ptr, size_t data_size, int flags) = 0;

  // Expose from hipc::FullPtr
  virtual Bulk Expose(const hipc::FullPtr<char>& ptr, size_t data_size, int flags) = 0;

  template<typename MetaT>
  Event* Send(MetaT &meta);
};

class Server {
 public:
  virtual ~Server() = default;

  // Expose from raw pointer
  virtual Bulk Expose(char* data, size_t data_size, int flags) = 0;

  // Expose from hipc::Pointer
  virtual Bulk Expose(const hipc::Pointer& ptr, size_t data_size, int flags) = 0;

  // Expose from hipc::FullPtr
  virtual Bulk Expose(const hipc::FullPtr<char>& ptr, size_t data_size, int flags) = 0;

  template<typename MetaT>
  Event* RecvMetadata(MetaT &meta);

  template<typename MetaT>
  Event* RecvBulks(MetaT &meta);

  virtual std::string GetAddress() const = 0;
};

// --- Transport Enum ---
enum class Transport { kZeroMq };

// --- Factory ---
class TransportFactory {
 public:
  // Users must include the correct transport header before calling these.
  static std::unique_ptr<Client> GetClient(const std::string& addr, Transport t,
                                          const std::string& protocol = "",
                                          int port = 0);
  static std::unique_ptr<Client> GetClient(const std::string& addr, Transport t,
                                          const std::string& protocol, int port,
                                          const std::string& domain);
  static std::unique_ptr<Server> GetServer(const std::string& addr, Transport t,
                                          const std::string& protocol = "",
                                          int port = 0);
  static std::unique_ptr<Server> GetServer(const std::string& addr, Transport t,
                                          const std::string& protocol, int port,
                                          const std::string& domain);
};

}  // namespace hshm::lbm