#include <hermes_shm/lightbeam/lightbeam.h>
#include <hermes_shm/lightbeam/transport_factory_impl.h>
#include <cassert>
#include <iostream>
#include <fstream>
#include <vector>
#include <thread>
#include <chrono>
#include <sys/stat.h>
#include <unistd.h>
#include <mpi.h>

using namespace hshm::lbm;

std::vector<std::string> ReadHosts(const std::string &hostfile) {
    std::vector<std::string> hosts;
    std::ifstream in(hostfile);
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty()) hosts.push_back(line);
    }
    return hosts;
}

Transport ParseTransport(const std::string &s) {
    if (s == "zeromq") return Transport::kZeroMq;
    if (s == "thallium") return Transport::kThallium;
    throw std::runtime_error("Unknown transport type: " + s);
}

void Clients(std::vector<std::unique_ptr<Client>> &clients, const std::string &magic) {
    for (auto &client : clients) {
        Bulk bulk = client->Expose(magic.data(), magic.size(), 0);
        Event *event = client->Send(bulk);
        while (!event->is_done) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        assert(event->error_code == 0);
        delete event;
    }
}

void ServerThread(Server &server, size_t num_clients, const std::string &magic) {
    for (size_t i = 0; i < num_clients; ++i) {
        std::vector<char> y(magic.size());
        Bulk bulk = server.Expose(y.data(), y.size(), 0);
        Event *event = nullptr;
        while (!event || !event->is_done) {
            if (event) delete event;
            event = server.Recv(bulk);
            if (!event->is_done) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
        assert(event->error_code == 0);
        std::string received(bulk.data, bulk.size);
        std::cout << "[Server] Received: " << received << std::endl;
        assert(received == magic);
        delete event;
    }
}

std::string WaitForServerAddr(const std::string &filename) {
    // Wait for the file to appear
    for (int i = 0; i < 100; ++i) {
        struct stat buffer;
        if (stat(filename.c_str(), &buffer) == 0) {
            std::ifstream in(filename);
            std::string addr;
            std::getline(in, addr);
            return addr;
        }
        usleep(100000); // 100ms
    }
    throw std::runtime_error("Timeout waiting for server address file: " + filename);
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int my_rank = 0, world_size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <zeromq|thallium> <hostfile> [protocol] [domain] [port]\n";
        std::cerr << "Note: Number of MPI processes (mpirun -n) should match the number of hosts in the hostfile." << std::endl;
        MPI_Finalize();
        return 1;
    }
    std::string transport_str = argv[1];
    std::string hostfile = argv[2];
    std::string magic = "1212412";

    // Parse optional protocol, domain, port
    std::string protocol = (transport_str == "zeromq") ? "tcp" : "ofi+sockets";
    std::string domain = "";
    int port = (transport_str == "zeromq") ? 8192 : 8200;
    if (argc > 3) protocol = argv[3];
    if (argc > 4) domain = argv[4];
    if (argc > 5) port = std::stoi(argv[5]);

    Transport transport = ParseTransport(transport_str);
    std::vector<std::string> hosts = ReadHosts(hostfile);
    if ((int)hosts.size() != world_size) {
        std::cerr << "Error: Number of MPI processes (" << world_size << ") does not match number of hosts in hostfile (" << hosts.size() << ")." << std::endl;
        MPI_Finalize();
        return 1;
    }

    bool is_server = (my_rank == 0);
    std::string server_addr = hosts[0];
    const std::string addrfile = "/tmp/thallium_server_addr.txt";

    if (transport == Transport::kThallium) {
        if (is_server) {
            auto server_ptr = TransportFactory::GetServer(server_addr, transport, protocol, port, domain);
            std::string actual_addr = server_ptr->GetAddress();
            std::ofstream(addrfile) << actual_addr << std::endl;
            std::thread server_thread(ServerThread, std::ref(*server_ptr), hosts.size() - 1, std::ref(magic));
            server_thread.join();
            std::cout << "[Server] All messages received!" << std::endl;
        }
        MPI_Barrier(MPI_COMM_WORLD);
        if (!is_server) {
            std::string actual_addr = WaitForServerAddr(addrfile);
            auto client_ptr = TransportFactory::GetClient(actual_addr, transport, protocol, port, domain);
            std::vector<std::unique_ptr<Client>> clients;
            clients.emplace_back(std::move(client_ptr));
            std::thread client_thread(Clients, std::ref(clients), std::ref(magic));
            client_thread.join();
            std::cout << "[Client] Message sent!" << std::endl;
        }
    } else {
        if (is_server) {
            auto server_ptr = TransportFactory::GetServer(server_addr, transport, protocol, port);
            std::thread server_thread(ServerThread, std::ref(*server_ptr), hosts.size() - 1, std::ref(magic));
            server_thread.join();
            std::cout << "[Server] All messages received!" << std::endl;
        }
        MPI_Barrier(MPI_COMM_WORLD);
        if (!is_server) {
            auto client_ptr = TransportFactory::GetClient(server_addr, transport, protocol, port);
            std::vector<std::unique_ptr<Client>> clients;
            clients.emplace_back(std::move(client_ptr));
            std::thread client_thread(Clients, std::ref(clients), std::ref(magic));
            client_thread.join();
            std::cout << "[Client] Message sent!" << std::endl;
        }
    }
    MPI_Finalize();
    return 0;
} 