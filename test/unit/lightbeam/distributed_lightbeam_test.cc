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
    int num_msgs = 10; // default
    int msg_size = 32; // default small message
    if (argc < 6) {
        std::cerr << "Usage: " << argv[0] << " <zeromq|thallium|libfabric> <hostfile> <protocol> <domain> <port> [num_msgs] [msg_size]\n";
        std::cerr << "All parameters are required except [num_msgs] and [msg_size]. Number of MPI processes (mpirun -n) should match the number of hosts in the hostfile." << std::endl;
        MPI_Finalize();
        return 1;
    }
    if (argc > 6) num_msgs = std::stoi(argv[6]);
    if (argc > 7) msg_size = std::stoi(argv[7]);
    std::string transport_str = argv[1];
    std::string hostfile = argv[2];
    std::string protocol = argv[3];
    std::string domain = argv[4];
    int port = std::stoi(argv[5]);
    std::string magic(msg_size, 'x');

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
    std::string actual_addr = server_ptr->GetAddress();
    std::cout << "[Rank " << my_rank << "] Server address: " << actual_addr << ", port: " << my_port << std::endl;
    // Start timing before any send
    auto global_start = std::chrono::high_resolution_clock::now();
    // Start server thread with num_msgs
    std::thread server_thread([&](){
        std::ostringstream oss;
        oss << std::this_thread::get_id();
        std::cout << "[ServerThread] Thread ID: " << oss.str() << std::endl;
        int received = 0;
        for (int i = 0; i < num_msgs * world_size; ++i) {
            auto recv_time = std::chrono::high_resolution_clock::now();
            std::vector<char> y(msg_size);
            Bulk bulk = server_ptr->Expose(y.data(), y.size(), 0);
            Event *event = nullptr;
            while (!event || !event->is_done) {
                if (event) delete event;
                event = server_ptr->Recv(bulk);
                if (!event->is_done) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
            }
            received++;
            delete event;
            double t = std::chrono::duration<double>(recv_time - global_start).count();
            std::cout << "[Rank " << my_rank << "] Received message " << received << " at " << t << " s" << std::endl;
        }
        auto end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(end - global_start).count();
        std::cout << "[Server] Received " << received << " messages. Time: " << elapsed << " s" << std::endl;
        std::cout << "[ServerThread] Exiting after receiving all messages" << std::endl;
    });

    MPI_Barrier(MPI_COMM_WORLD);
    // Gather all server addresses using MPI_Allgather
    const int addr_len = 256;
    std::vector<char> addr_buf(addr_len, 0);
    strncpy(addr_buf.data(), actual_addr.c_str(), addr_len - 1);
    std::vector<char> all_addrs(world_size * addr_len, 0);
    MPI_Allgather(addr_buf.data(), addr_len, MPI_CHAR, all_addrs.data(), addr_len, MPI_CHAR, MPI_COMM_WORLD);
    std::vector<std::string> server_addrs;
    for (int i = 0; i < world_size; ++i) {
        server_addrs.emplace_back(&all_addrs[i * addr_len]);
    }
    std::vector<std::unique_ptr<Client>> clients;
    for (int i = 0; i < world_size; ++i) {
        auto client_ptr = TransportFactory::GetClient(server_addrs[i], transport, protocol, 0, "");
        clients.emplace_back(std::move(client_ptr));
    }
    int sent = 0;
    for (int m = 0; m < num_msgs; ++m) {
        for (size_t i = 0; i < clients.size(); ++i) {
            auto send_time = std::chrono::high_resolution_clock::now();
            Bulk bulk = clients[i]->Expose(magic.data(), magic.size(), 0);
            Event *event = clients[i]->Send(bulk);
            while (!event->is_done) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            assert(event->error_code == 0);
            delete event;
            sent++;
            double t = std::chrono::duration<double>(send_time - global_start).count();
            std::cout << "[Rank " << my_rank << "] Sent message " << sent << " to server " << i << " at " << t << " s" << std::endl;
        }
    }
    server_thread.join();
    auto global_end = std::chrono::high_resolution_clock::now();
    double global_elapsed = std::chrono::duration<double>(global_end - global_start).count();
    std::cout << "[Rank " << my_rank << "] All server messages received!" << std::endl;
    std::cout << "[Rank " << my_rank << "] Overall runtime (first send to last receive): " << global_elapsed << " s" << std::endl;
    MPI_Finalize();
    return 0;
} 