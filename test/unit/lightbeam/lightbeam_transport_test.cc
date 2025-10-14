#include <hermes_shm/lightbeam/zmq_transport.h>
#include <cassert>
#include <iostream>
#include <vector>
#include <thread>
#include <chrono>

using namespace hshm::lbm;

void TestZeroMQ() {
#ifdef HSHM_ENABLE_ZMQ
  std::cout << "\n==== Testing ZeroMQ ====\n";

  std::string addr = "127.0.0.1";
  std::string protocol = "tcp";
  int port = 8192;

  auto server = std::make_unique<ZeroMqServer>(addr, protocol, port);
  auto client = std::make_unique<ZeroMqClient>(addr, protocol, port);

  // Give ZMQ time to connect
  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  const std::string magic = "unit_test_magic";

  // Client creates metadata and sends
  LbmMeta send_meta;
  Bulk send_bulk = client->Expose(magic.data(), magic.size(), 0);
  send_meta.bulks.push_back(send_bulk);

  Event* send_event = client->Send(send_meta);
  assert(send_event->is_done);
  assert(send_event->error_code == 0);
  std::cout << "Client sent " << send_event->bytes_transferred << " bytes\n";
  delete send_event;

  // Server receives metadata
  LbmMeta recv_meta;
  Event* recv_event = nullptr;
  while (!recv_event || !recv_event->is_done) {
    if (recv_event) delete recv_event;
    recv_event = server->RecvMetadata(recv_meta);
    if (!recv_event->is_done) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }
  assert(recv_event->error_code == 0);
  assert(recv_meta.bulks.size() == 1);
  delete recv_event;

  // Allocate buffer and receive bulks
  std::vector<char> recv_buf(recv_meta.bulks[0].size);
  recv_meta.bulks[0] = server->Expose(recv_buf.data(), recv_buf.size(), 0);

  recv_event = server->RecvBulks(recv_meta);
  while (!recv_event->is_done) {
    delete recv_event;
    recv_event = server->RecvBulks(recv_meta);
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  assert(recv_event->error_code == 0);
  delete recv_event;

  std::string received(recv_buf.begin(), recv_buf.end());
  std::cout << "Received: " << received << std::endl;
  assert(received == magic);

  std::cout << "[ZeroMQ] Test passed!\n";
#else
  std::cout << "ZeroMQ not enabled, skipping test\n";
#endif
}

int main() {
  TestZeroMQ();
  std::cout << "\nAll transport tests passed!" << std::endl;
  return 0;
} 