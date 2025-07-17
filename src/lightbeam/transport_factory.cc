#include "hermes_shm/lightbeam/transport_factory.h"

namespace hshm::lbm {

std::unique_ptr<IServer> Transport::CreateServer(TransportType type) {
  // Transport factory is no longer used - libfabric is used directly
  return nullptr;
}

std::unique_ptr<IClient> Transport::CreateClient(TransportType type) {
  // Transport factory is no longer used - libfabric is used directly
  return nullptr;
}

}  // namespace hshm::lbm 