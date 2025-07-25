#pragma once
// Common types, interfaces, and factory for lightbeam transports.
// Users must include the appropriate transport header (zmq_transport.h or
// thallium_transport.h) before using the factory for that transport.
#include <string>
#include <memory>
#include <queue>
#include <mutex>
#include <cstring>
#include <cassert>
#include <iostream>

namespace hshm::lbm {

// --- Types ---
struct Event {
  bool is_done = false;
  int error_code = 0;
  std::string error_message;
  size_t bytes_transferred = 0;
};

struct Bulk {
  char* data;
  size_t size;
  int flags;
  void* desc = nullptr;  // For RDMA memory registration
  void* mr = nullptr;    // For RDMA memory region handle (fid_mr*)
};

// --- Interfaces ---
class Client {
 public:
  virtual ~Client() = default;
  virtual Bulk Expose(const char* data, size_t data_size, int flags) = 0;
  virtual Event* Send(const Bulk& bulk) = 0;
};

class Server {
 public:
  virtual ~Server() = default;
  virtual Bulk Expose(char* data, size_t data_size, int flags) = 0;
  virtual Event* Recv(const Bulk& bulk) = 0;
  virtual std::string GetAddress() const = 0;
};

// --- Transport Enum ---
enum class Transport { kZeroMq, kThallium, kLibfabric };

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