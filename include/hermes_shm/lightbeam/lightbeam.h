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
#include "hermes_shm/types/bitfield.h"

namespace hshm::lbm {

// --- Bulk Flags ---
#define BULK_EXPOSE BIT_OPT(hshm::u32, 0)  // Bulk is exposed (sender exposes for reading)
#define BULK_WRITE BIT_OPT(hshm::u32, 1)   // Bulk is exposed for writing (receiver)

// --- Types ---
struct Bulk {
  hipc::FullPtr<char> data;
  size_t size;
  hshm::bitfield32_t flags;  // BULK_READ or BULK_WRITE
  void* desc = nullptr;      // For RDMA memory registration
  void* mr = nullptr;        // For RDMA memory region handle (fid_mr*)
};

// --- Metadata Base Class ---
class LbmMeta {
 public:
  std::vector<Bulk> send;  // Bulks marked BULK_WRITE (sender side)
  std::vector<Bulk> recv;  // Bulks marked BULK_EXPOSE (receiver side)
};

// --- Interfaces ---
class Client {
 public:
  virtual ~Client() = default;

  // Expose from raw pointer
  virtual Bulk Expose(const char* data, size_t data_size, u32 flags) = 0;

  // Expose from hipc::Pointer
  virtual Bulk Expose(const hipc::Pointer& ptr, size_t data_size, u32 flags) = 0;

  // Expose from hipc::FullPtr
  virtual Bulk Expose(const hipc::FullPtr<char>& ptr, size_t data_size, u32 flags) = 0;

  template<typename MetaT>
  int Send(MetaT &meta);
};

class Server {
 public:
  virtual ~Server() = default;

  // Expose from raw pointer
  virtual Bulk Expose(char* data, size_t data_size, u32 flags) = 0;

  // Expose from hipc::Pointer
  virtual Bulk Expose(const hipc::Pointer& ptr, size_t data_size, u32 flags) = 0;

  // Expose from hipc::FullPtr
  virtual Bulk Expose(const hipc::FullPtr<char>& ptr, size_t data_size, u32 flags) = 0;

  template<typename MetaT>
  int RecvMetadata(MetaT &meta);

  template<typename MetaT>
  int RecvBulks(MetaT &meta);

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