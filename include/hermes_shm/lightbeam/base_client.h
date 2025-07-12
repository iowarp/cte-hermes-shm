#pragma once
#include "hermes_shm/lightbeam/types.h"
#include <memory>

namespace hshm::lbm {

class IClient {
public:
  virtual void Connect(const std::string &url, TransportType transport = TransportType::AUTO) = 0;
  virtual void Disconnect(const std::string &url) = 0;
  virtual Bulk Expose(const std::string &url, const char *data, size_t data_size, int flags) = 0;
  
  // Return unique_ptr to match existing headers
  virtual std::unique_ptr<Event> Send(const Bulk &bulk) = 0;
  virtual std::unique_ptr<Event> Recv(char *buffer, size_t buffer_size, const std::string &from_url) = 0;
  
  // Add timeout parameter to match existing headers
  virtual void ProcessCompletions(double timeout_msec = 0.0) = 0;
  
  virtual ~IClient() = default;
};

} // namespace hshm::lbm