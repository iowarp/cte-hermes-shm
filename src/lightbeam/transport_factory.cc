#include "hermes_shm/lightbeam/transport_factory.h"
#include "hermes_shm/lightbeam/zmq/server.h"
#include "hermes_shm/lightbeam/zmq/client.h"
#include "hermes_shm/lightbeam/libfabric/server.h"
#include "hermes_shm/lightbeam/libfabric/client.h"

namespace hshm::lbm {

std::unique_ptr<IServer> Transport::CreateServer(TransportType type) {
  switch (type) {
    case TransportType::TCP:
      return std::make_unique<zmq::Server>();
    case TransportType::RDMA:
      return std::make_unique<libfabric::Server>();
    default:
      return nullptr;
  }
}

std::unique_ptr<IClient> Transport::CreateClient(TransportType type) {
  switch (type) {
    case TransportType::TCP:
      return std::make_unique<zmq::Client>();
    case TransportType::RDMA:
      return std::make_unique<libfabric::Client>();
    default:
      return nullptr;
  }
}

}  // namespace hshm::lbm 