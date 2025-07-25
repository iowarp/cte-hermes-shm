#include <hermes_shm/lightbeam/lightbeam.h>
#include <hermes_shm/lightbeam/transport_factory_impl.h>
#include <cassert>
#include <iostream>
#include <vector>
#include <thread>
#include <chrono>

using namespace hshm::lbm;

class LightbeamTransportTest {
 public:
  LightbeamTransportTest(Transport transport, const std::string& addr,
                         const std::string& protocol, int port)
      : transport_(transport),
        addr_(addr),
        protocol_(protocol),
        port_(port) {}

  void Run() {
    std::cout << "\n==== Testing backend: " << BackendName() << " ====\n";
    auto server_ptr =
        TransportFactory::GetServer(addr_, transport_, protocol_, port_);
    std::string server_addr = server_ptr->GetAddress();
    std::unique_ptr<Client> client_ptr;
    if (transport_ == Transport::kLibfabric) {
      client_ptr = TransportFactory::GetClient(server_addr, transport_,
                                               protocol_, port_);
    } else {
      client_ptr = TransportFactory::GetClient(server_addr, transport_,
                                               protocol_, port_);
    }

    const std::string magic = "unit_test_magic";
    // Client exposes and sends data
    Bulk send_bulk = client_ptr->Expose(magic.data(), magic.size(), 0);
    Event* send_event = client_ptr->Send(send_bulk);
    while (!send_event->is_done) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    assert(send_event->error_code == 0);
    delete send_event;

    // Server exposes buffer and receives data
    std::vector<char> recv_buf(magic.size());
    Bulk recv_bulk = server_ptr->Expose(recv_buf.data(), recv_buf.size(), 0);
    Event* recv_event = nullptr;
    while (!recv_event || !recv_event->is_done) {
      if (recv_event) delete recv_event;
      recv_event = server_ptr->Recv(recv_bulk);
      if (!recv_event->is_done) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
    }
    assert(recv_event->error_code == 0);
    std::string received(recv_bulk.data, recv_bulk.size);
    std::cout << "Received: " << received << std::endl;
    assert(received == magic);
    delete recv_event;
    std::cout << "[" << BackendName() << "] Test passed!\n";
  }

 private:
  std::string BackendName() const {
    switch (transport_) {
      case Transport::kZeroMq:
        return "ZeroMQ";
      case Transport::kThallium:
        return "Thallium";
      case Transport::kLibfabric:
        return "Libfabric";
      default:
        return "Unknown";
    }
  }
  Transport transport_;
  std::string addr_;
  std::string protocol_;
  int port_;
};

int main() {
  // Test ZeroMQ
#ifdef HSHM_ENABLE_ZMQ
  {
    std::string zmq_addr = "127.0.0.1";
    std::string zmq_protocol = "tcp";
    int zmq_port = 8192;
    LightbeamTransportTest test(Transport::kZeroMq, zmq_addr, zmq_protocol,
                                zmq_port);
    test.Run();
  }
#endif
  // Test Thallium
  {
    std::string thallium_addr = "127.0.0.1";
    std::string thallium_protocol = "ofi+sockets";
    int thallium_port = 8193;
    LightbeamTransportTest test(Transport::kThallium, thallium_addr,
                                thallium_protocol, thallium_port);
    test.Run();
  }
  // Test Libfabric
  {
    std::string libfabric_addr = "127.0.0.1";
    std::string libfabric_protocol = "tcp";
    int libfabric_port = 9222;
    LightbeamTransportTest test(Transport::kLibfabric, libfabric_addr,
                                libfabric_protocol, libfabric_port);
    test.Run();
  }
  std::cout << "All transport tests passed!" << std::endl;
  return 0;
} 