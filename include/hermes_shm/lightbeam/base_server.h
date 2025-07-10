#pragma once
#include <string>
#include "hermes_shm/lightbeam/types.h"

namespace hshm::lbm {

class IServer {
 public:
  virtual void StartServer(const std::string &url, TransportType transport) = 0;
  virtual void Stop() = 0;
  virtual void ProcessMessages() = 0;
  virtual bool IsRunning() const = 0;
  virtual ~IServer() = default;
};

}  // namespace hshm::lbm 