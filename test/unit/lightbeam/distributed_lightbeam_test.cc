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
#include <sstream>
#include <ifaddrs.h>
#include <arpa/inet.h>
#include <net/if.h>
#include <netdb.h>

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
    if (s == "libfabric") return Transport::kLibfabric;
    throw std::runtime_error("Unknown transport type: " + s);
}

void Clients(std::vector<std::unique_ptr<Client>> &clients, const std::string &magic) {
    int my_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    std::ostringstream oss;
    oss << std::this_thread::get_id();
    std::cout << "[Rank " << my_rank << "] [Clients] Thread ID: " << oss.str() << std::endl;
    for (size_t i = 0; i < clients.size(); ++i) {
        std::cout << "[Rank " << my_rank << "] [Clients] Sending to server " << i << std::endl;
        Bulk bulk = clients[i]->Expose(magic.data(), magic.size(), 0);
        Event *event = clients[i]->Send(bulk);
        while (!event->is_done) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        std::cout << "[Rank " << my_rank << "] [Clients] Sent to server " << i << ", error_code=" << event->error_code << std::endl;
        assert(event->error_code == 0);
        delete event;
    }
}

void ServerThread(Server &server, size_t num_clients, const std::string &magic) {
    std::ostringstream oss;
    oss << std::this_thread::get_id();
    std::cout << "[ServerThread] Thread ID: " << oss.str() << std::endl;
    for (size_t i = 0; i < num_clients; ++i) {
        std::cout << "[Server] Waiting for message " << i << std::endl;
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
        std::cout << "[Server] Received message " << i << ", error_code=" << event->error_code << std::endl;
        assert(event->error_code == 0);
        std::string received(bulk.data, bulk.size);
        std::cout << "[Server] Received: " << received << std::endl;
        assert(received == magic);
        delete event;
    }
    std::cout << "[ServerThread] Exiting after receiving all messages" << std::endl;
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

std::string get_primary_ip() {
    struct ifaddrs *ifaddr, *ifa;
    char ip[INET_ADDRSTRLEN];
    std::string result;
    getifaddrs(&ifaddr);
    for (ifa = ifaddr; ifa != NULL; ifa = ifa->ifa_next) {
        if (ifa->ifa_addr && ifa->ifa_addr->sa_family == AF_INET &&
            !(ifa->ifa_flags & IFF_LOOPBACK) && (ifa->ifa_flags & IFF_UP)) {
            void* addr_ptr = &((struct sockaddr_in*)ifa->ifa_addr)->sin_addr;
            inet_ntop(AF_INET, addr_ptr, ip, INET_ADDRSTRLEN);
            result = ip;
            break;
        }
    }
    freeifaddrs(ifaddr);
    return result;
}

void PrintAllInterfaces() {
    struct ifaddrs *ifaddr, *ifa;
    char host[NI_MAXHOST];
    if (getifaddrs(&ifaddr) == -1) {
        perror("getifaddrs");
        return;
    }
    freeifaddrs(ifaddr);
}

int main(int argc, char **argv) {
    // PrintAllInterfaces();
    MPI_Init(&argc, &argv);
    int my_rank = 0, world_size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (argc < 6) {
        std::cerr << "Usage: " << argv[0] << " <zeromq|thallium|libfabric> <hostfile> <protocol> <domain> <port>\n";
        std::cerr << "All parameters are required. Number of MPI processes (mpirun -n) should match the number of hosts in the hostfile." << std::endl;
        MPI_Finalize();
        return 1;
    }
    std::string transport_str = argv[1];
    std::string hostfile = argv[2];
    std::string protocol = argv[3];
    std::string domain = argv[4];
    int port = std::stoi(argv[5]);
    std::string magic = "1212412";

    Transport transport = ParseTransport(transport_str);
    std::vector<std::string> hosts = ReadHosts(hostfile);
    if ((int)hosts.size() != world_size) {
        std::cerr << "Error: Number of MPI processes (" << world_size << ") does not match number of hosts in hostfile (" << hosts.size() << ")." << std::endl;
        MPI_Finalize();
        return 1;
    }

    int my_port = (transport == Transport::kThallium) ? 0 : port + my_rank;
    std::string bind_addr = get_primary_ip();

    std::string domain_arg = (transport == Transport::kThallium) ? "" : domain;
    auto server_ptr = TransportFactory::GetServer(bind_addr, transport, protocol, my_port, domain_arg);
    // printf("[Debug] server_ptr->GetAddress(): %s\n", server_ptr->GetAddress().c_str());
    std::string actual_addr = server_ptr->GetAddress();
    std::cout << "[Rank " << my_rank << "] Server address: " << actual_addr << ", port: " << my_port << std::endl;
    std::thread server_thread(ServerThread, std::ref(*server_ptr), world_size, std::ref(magic));

    // Synchronize all ranks to ensure all servers are ready before clients connect
    MPI_Barrier(MPI_COMM_WORLD);

    // Gather all server addresses using MPI_Allgather
    const int addr_len = 256;
    std::vector<char> addr_buf(addr_len, 0);
    strncpy(addr_buf.data(), actual_addr.c_str(), addr_len - 1);
    std::vector<char> all_addrs(world_size * addr_len, 0);
    MPI_Allgather(addr_buf.data(), addr_len, MPI_CHAR, all_addrs.data(), addr_len, MPI_CHAR, MPI_COMM_WORLD);

    // Build vector of all addresses (for client connections)
    std::vector<std::string> server_addrs;
    for (int i = 0; i < world_size; ++i) {
        server_addrs.emplace_back(&all_addrs[i * addr_len]);
    }

    // All ranks run the client logic (including rank 0)
    std::vector<std::unique_ptr<Client>> clients;
    for (int i = 0; i < world_size; ++i) {
        // Use the full address string as returned by the server, protocol/port/domain are ignored for the client
        auto client_ptr = TransportFactory::GetClient(server_addrs[i], transport, protocol, 0, "");
        clients.emplace_back(std::move(client_ptr));
    }
    std::thread client_thread(Clients, std::ref(clients), std::ref(magic));
    client_thread.join();
    std::cout << "[Rank " << my_rank << "] All client messages sent!" << std::endl;
    server_thread.join();
    std::cout << "[Rank " << my_rank << "] All server messages received!" << std::endl;
    MPI_Finalize();
    return 0;
} 