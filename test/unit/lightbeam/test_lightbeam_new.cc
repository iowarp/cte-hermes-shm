#include <hermes_shm/lightbeam/zmq_transport.h>
#include <cassert>
#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <cstring>

using namespace hshm::lbm;

// Custom metadata class that inherits from LbmMeta
class TestMeta : public LbmMeta {
 public:
  int request_id;
  std::string operation;
};

// Cereal serialization for TestMeta
namespace cereal {
  template<class Archive>
  void serialize(Archive& ar, TestMeta& meta) {
    ar(meta.bulks, meta.request_id, meta.operation);
  }
}

void TestBasicTransfer() {
  std::cout << "\n==== Testing Basic Transfer with New API ====\n";

#ifdef HSHM_ENABLE_ZMQ
  // Create server
  std::string addr = "127.0.0.1";
  std::string protocol = "tcp";
  int port = 8195;

  auto server = std::make_unique<ZeroMqServer>(addr, protocol, port);
  auto client = std::make_unique<ZeroMqClient>(addr, protocol, port);

  // Give ZMQ time to connect
  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  // Prepare data on client side
  const char* data1 = "Hello, World!";
  const char* data2 = "Testing Lightbeam";
  size_t size1 = strlen(data1);
  size_t size2 = strlen(data2);

  // Create metadata and expose bulks
  TestMeta send_meta;
  send_meta.request_id = 42;
  send_meta.operation = "test_op";

  Bulk bulk1 = client->Expose(data1, size1, 0);
  Bulk bulk2 = client->Expose(data2, size2, 0);

  send_meta.bulks.push_back(bulk1);
  send_meta.bulks.push_back(bulk2);

  // Send metadata and bulks
  Event* send_event = client->Send(send_meta);
  assert(send_event->is_done);
  assert(send_event->error_code == 0);
  std::cout << "Client sent " << send_event->bytes_transferred << " bytes\n";
  delete send_event;

  // Server receives metadata
  TestMeta recv_meta;
  Event* recv_event = nullptr;
  while (!recv_event || !recv_event->is_done) {
    if (recv_event) delete recv_event;
    recv_event = server->RecvMetadata(recv_meta);
    if (!recv_event->is_done) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }
  assert(recv_event->error_code == 0);
  std::cout << "Server received metadata: request_id=" << recv_meta.request_id
            << ", operation=" << recv_meta.operation << "\n";
  assert(recv_meta.request_id == 42);
  assert(recv_meta.operation == "test_op");
  assert(recv_meta.bulks.size() == 2);
  delete recv_event;

  // Allocate buffers for receiving bulks
  std::vector<char> recv_buf1(recv_meta.bulks[0].size);
  std::vector<char> recv_buf2(recv_meta.bulks[1].size);

  recv_meta.bulks[0] = server->Expose(recv_buf1.data(), recv_buf1.size(), 0);
  recv_meta.bulks[1] = server->Expose(recv_buf2.data(), recv_buf2.size(), 0);

  // Receive bulks
  recv_event = server->RecvBulks(recv_meta);
  while (!recv_event->is_done) {
    delete recv_event;
    recv_event = server->RecvBulks(recv_meta);
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  assert(recv_event->error_code == 0);
  std::cout << "Server received " << recv_event->bytes_transferred << " bytes of bulk data\n";
  delete recv_event;

  // Verify received data
  std::string received1(recv_buf1.begin(), recv_buf1.end());
  std::string received2(recv_buf2.begin(), recv_buf2.end());

  std::cout << "Bulk 1: " << received1 << "\n";
  std::cout << "Bulk 2: " << received2 << "\n";

  assert(received1 == data1);
  assert(received2 == data2);

  std::cout << "[New API] Test passed!\n";
#else
  std::cout << "ZMQ not enabled, skipping test\n";
#endif
}

void TestMultipleBulks() {
  std::cout << "\n==== Testing Multiple Bulks Transfer ====\n";

#ifdef HSHM_ENABLE_ZMQ
  std::string addr = "127.0.0.1";
  std::string protocol = "tcp";
  int port = 8196;

  auto server = std::make_unique<ZeroMqServer>(addr, protocol, port);
  auto client = std::make_unique<ZeroMqClient>(addr, protocol, port);

  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  // Prepare multiple data chunks
  std::vector<std::string> data_chunks = {
    "Chunk 1",
    "Chunk 2 is longer",
    "Chunk 3",
    "Final chunk 4"
  };

  // Create metadata
  LbmMeta send_meta;
  for (const auto& chunk : data_chunks) {
    Bulk bulk = client->Expose(chunk.data(), chunk.size(), 0);
    send_meta.bulks.push_back(bulk);
  }

  // Send
  Event* send_event = client->Send(send_meta);
  assert(send_event->is_done && send_event->error_code == 0);
  delete send_event;

  // Receive metadata
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
  assert(recv_meta.bulks.size() == data_chunks.size());
  delete recv_event;

  // Allocate buffers and receive bulks
  std::vector<std::vector<char>> recv_buffers;
  for (size_t i = 0; i < recv_meta.bulks.size(); ++i) {
    recv_buffers.emplace_back(recv_meta.bulks[i].size);
    recv_meta.bulks[i] = server->Expose(recv_buffers[i].data(),
                                        recv_buffers[i].size(), 0);
  }

  recv_event = server->RecvBulks(recv_meta);
  while (!recv_event->is_done) {
    delete recv_event;
    recv_event = server->RecvBulks(recv_meta);
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  assert(recv_event->error_code == 0);
  delete recv_event;

  // Verify all chunks
  for (size_t i = 0; i < data_chunks.size(); ++i) {
    std::string received(recv_buffers[i].begin(), recv_buffers[i].end());
    std::cout << "Chunk " << i << ": " << received << "\n";
    assert(received == data_chunks[i]);
  }

  std::cout << "[Multiple Bulks] Test passed!\n";
#else
  std::cout << "ZMQ not enabled, skipping test\n";
#endif
}

int main() {
  TestBasicTransfer();
  TestMultipleBulks();
  std::cout << "\nAll new API tests passed!" << std::endl;
  return 0;
}
