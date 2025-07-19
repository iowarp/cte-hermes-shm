#include <hermes_shm/lightbeam/lightbeam.h>
#include <hermes_shm/lightbeam/thallium_transport.h>
#include <thread>
#include <vector>
#include <cassert>
#include <iostream>
#include <chrono>

using namespace hshm::lbm;

const std::string magic = "1212412";

void Clients(Client &client, int num_iters) {
    for (int i = 0; i < num_iters; ++i) {
        Bulk bulk = client.Expose(magic.data(), magic.size(), 0);
        std::cout << "[Client] Sending: " << std::string(bulk.data, bulk.size) << std::endl;
        Event *event = client.Send(bulk);
        while (!event->is_done) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        std::cout << "[Client] Event done, error_code: " << event->error_code << std::endl;
        assert(event->error_code == 0);
    }
}

void ServerThread(Server &server, int num_iters) {
    for (int i = 0; i < num_iters; ++i) {
        std::vector<char> y(magic.size());
        Bulk bulk = server.Expose(y.data(), y.size(), 0);
        Event* recv_event = nullptr;
        while (!recv_event || !recv_event->is_done) {
            recv_event = server.Recv(bulk);
            if (!recv_event->is_done) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
        std::cout << "[Server] Received: " << std::string(bulk.data, bulk.size) << std::endl;
        assert(std::string(bulk.data, bulk.size) == magic);
    }
}

int main() {
    std::string url = "ofi+sockets://127.0.0.1:8192";
    int num_clients = 4;
    int num_iters = 10;

    auto server_ptr = TransportFactory::GetServer(url, Transport::kThallium);
    std::string server_addr = server_ptr->GetAddress();
    auto client_ptr = TransportFactory::GetClient(server_addr, Transport::kThallium);

    std::thread server_thread(ServerThread, std::ref(*server_ptr), num_iters);
    std::vector<std::thread> client_threads;
    for (int i = 0; i < num_clients; ++i) {
        client_threads.emplace_back(Clients, std::ref(*client_ptr), num_iters);
    }

    for (auto &t : client_threads) t.join();
    server_thread.join();

    std::cout << "Test passed!" << std::endl;
    return 0;
} 