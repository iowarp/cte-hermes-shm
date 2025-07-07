#include "hermes_shm/lightbeam/lightbeam.h"
#include <thread>
#include <iostream>
#include <cstring>
#include <chrono>
#include <memory>
#include <atomic>

using namespace hshm::lbm;

int main() {
    // Use smart pointers to avoid instantiating destructors in this compilation unit
    auto server = std::make_unique<Server>();
    std::atomic<bool> server_ready{false};
    std::atomic<bool> stop_server{false};
    
    std::thread server_thread([&server, &server_ready, &stop_server]{
        server->StartServer("tcp://127.0.0.1:5555");
        server_ready = true;
        std::cout << "Server started and ready" << std::endl;
        
        while (!stop_server) {
            server->ProcessMessages();
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        server->Stop();
        std::cout << "Server stopped" << std::endl;
    });
    
    // Wait for server to be ready
    while (!server_ready) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100)); // Extra wait for binding
    
    auto client = std::make_unique<Client>();
    client->Connect("tcp://127.0.0.1:5555");
    std::cout << "Client connected" << std::endl;
    
    // Give some time for connection to establish
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    const char msg[] = "hello rajni";
    std::cout << "Sending message: " << msg << std::endl;
    
    Bulk bulk = client->Expose("tcp://127.0.0.1:5555", msg, strlen(msg), 0);
    Event* ev = client->Send(bulk);
    
    if (ev) {
        std::cout << "Send event - done: " << ev->is_done 
                  << ", error: " << ev->error_code 
                  << ", bytes: " << ev->bytes_transferred << std::endl;
    }
    
    // Give time for message to be processed
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    char recv_buf[32] = {};
    std::cout << "Attempting to receive..." << std::endl;
    Event* ev2 = client->Recv(recv_buf, sizeof(recv_buf) - 1, "tcp://127.0.0.1:5555");
    
    if (ev2) {
        std::cout << "Recv event - done: " << ev2->is_done 
                  << ", error: " << ev2->error_code 
                  << ", bytes: " << ev2->bytes_transferred << std::endl;
    }
    
    std::cout << "Sent: '" << msg << "', Received: '" << recv_buf << "'" << std::endl;
    
    client->Disconnect("tcp://127.0.0.1:5555");
    std::cout << "Client disconnected" << std::endl;
    
    stop_server = true;
    server_thread.join();
    
    return 0;
}