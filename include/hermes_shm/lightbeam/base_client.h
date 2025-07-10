#pragma once
#include <string>
#include "hermes_shm/lightbeam/types.h"

namespace hshm::lbm {

class IClient {
 public:
  virtual void Connect(const std::string &url, TransportType transport) = 0;
  virtual void Disconnect(const std::string &url) = 0;
  virtual Bulk Expose(const std::string &url, const char *data, size_t data_size, int flags) = 0;
  virtual Event* Send(const Bulk &bulk) = 0;
  virtual Event* Recv(char *buffer, size_t buffer_size, const std::string &from_url) = 0;
  virtual void ProcessCompletions() = 0;
  virtual ~IClient() = default;
};

}  // namespace hshm::lbm 