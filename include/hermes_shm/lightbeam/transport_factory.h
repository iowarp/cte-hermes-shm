#pragma once
#include <memory>
#include "hermes_shm/lightbeam/base_server.h"
#include "hermes_shm/lightbeam/base_client.h"
#include "hermes_shm/lightbeam/lightbeam.h"

namespace hshm::lbm {

class Transport {
 public:
  static std::unique_ptr<IServer> CreateServer(TransportType type);
  static std::unique_ptr<IClient> CreateClient(TransportType type);
};

}  // namespace hshm::lbm 