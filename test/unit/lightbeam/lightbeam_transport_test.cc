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
    LightbeamTransportTest(Transport transport, const std::string& url)
        : transport_(transport), url_(url) {}

    void Run() {
        std::cout << "\n==== Testing backend: " << BackendName() << " ====" << std::endl;
        auto server_ptr = TransportFactory::GetServer(url_, transport_);
        std::string server_addr = server_ptr->GetAddress();
        auto client_ptr = TransportFactory::GetClient(server_addr, transport_);

        const std::string magic = "unit_test_magic";
        // Client exposes and sends data
        Bulk send_bulk = client_ptr->Expose(magic.data(), magic.size(), 0);
        Event *send_event = client_ptr->Send(send_bulk);
        while (!send_event->is_done) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        assert(send_event->error_code == 0);
        delete send_event;

        // Server exposes buffer and receives data
        std::vector<char> recv_buf(magic.size());
        Bulk recv_bulk = server_ptr->Expose(recv_buf.data(), recv_buf.size(), 0);
        Event *recv_event = nullptr;
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
            case Transport::kZeroMq: return "ZeroMQ";
            case Transport::kThallium: return "Thallium";
            default: return "Unknown";
        }
    }
    Transport transport_;
    std::string url_;
};

int main() {
    // Test ZeroMQ
#ifdef HSHM_ENABLE_ZMQ
    {
        std::string zmq_url = "tcp://127.0.0.1:8192";
        LightbeamTransportTest test(Transport::kZeroMq, zmq_url);
        test.Run();
    }
#endif
    // Test Thallium
    {
        std::string thallium_url = "ofi+sockets://127.0.0.1:8193";
        LightbeamTransportTest test(Transport::kThallium, thallium_url);
        test.Run();
    }
    std::cout << "All transport tests passed!" << std::endl;
    return 0;
} 