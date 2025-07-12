#include "hermes_shm/lightbeam/lightbeam.h"
#include "hermes_shm/util/timer.h"
#include <iostream>
#include <thread>
#include <chrono>
#include <vector>
#include <memory>
#include <cassert>
#include <random>

class NonBlockingTest {
private:
    std::vector<std::string> test_results;
    
    void log_result(const std::string& test_name, bool passed, const std::string& details = "") {
        std::string result = "[" + std::string(passed ? "PASS" : "FAIL") + "] " + test_name;
        if (!details.empty()) {
            result += " - " + details;
        }
        test_results.push_back(result);
        std::cout << result << std::endl;
    }
    
public:
    void test_nonblocking_send_recv() {
        std::cout << "\n=== Testing Non-blocking Send/Recv ===" << std::endl;
        
        auto server = hshm::lbm::Transport::CreateServer(hshm::lbm::TransportType::TCP);
        auto client = hshm::lbm::Transport::CreateClient(hshm::lbm::TransportType::TCP);
        
        std::string url = "tcp://127.0.0.1:5555";
        server->StartServer(url, hshm::lbm::TransportType::TCP);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        
        client->Connect(url, hshm::lbm::TransportType::TCP);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        
        // Test 1: Send should return immediately
        std::string message = "Test message";
        auto bulk = client->Expose(url, message.c_str(), message.length(), 0);
        
        auto start_time = std::chrono::high_resolution_clock::now();
        auto send_event = client->Send(bulk);
        auto end_time = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        bool send_nonblocking = duration.count() < 1000; // Should complete in < 1ms
        
        log_result("Send returns immediately", send_nonblocking, 
                  "Duration: " + std::to_string(duration.count()) + " μs");
        
        // Test 2: Recv should return immediately
        char recv_buffer[1024];
        start_time = std::chrono::high_resolution_clock::now();
        auto recv_event = client->Recv(recv_buffer, sizeof(recv_buffer), url);
        end_time = std::chrono::high_resolution_clock::now();
        
        duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        bool recv_nonblocking = duration.count() < 1000; // Should complete in < 1ms
        
        log_result("Recv returns immediately", recv_nonblocking,
                  "Duration: " + std::to_string(duration.count()) + " μs");
        
        // Test 3: Events should be unique heap-allocated objects
        auto send_event2 = client->Send(bulk);
        bool unique_events = (send_event.get() != send_event2.get()) && 
                            (send_event->event_id != send_event2->event_id);
        
        log_result("Events are unique heap-allocated objects", unique_events,
                  "Event1 ID: " + std::to_string(send_event->event_id) + 
                  ", Event2 ID: " + std::to_string(send_event2->event_id));
        
        // Cleanup
        client->Disconnect(url);
        server->Stop();
    }
    
    void test_multiple_concurrent_messages() {
        std::cout << "\n=== Testing Multiple Concurrent Messages ===" << std::endl;
        
        auto server = hshm::lbm::Transport::CreateServer(hshm::lbm::TransportType::TCP);
        auto client = hshm::lbm::Transport::CreateClient(hshm::lbm::TransportType::TCP);
        
        std::string url = "tcp://127.0.0.1:5556";
        server->StartServer(url, hshm::lbm::TransportType::TCP);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        
        client->Connect(url, hshm::lbm::TransportType::TCP);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        
        // Test: Send multiple messages without waiting for responses
        const int num_messages = 5;
        std::vector<std::unique_ptr<hshm::lbm::Event>> send_events;
        std::vector<std::unique_ptr<hshm::lbm::Event>> recv_events;
        std::vector<std::unique_ptr<char[]>> recv_buffers;
        
        // Send multiple messages
        for (int i = 0; i < num_messages; ++i) {
            std::string message = "Message " + std::to_string(i + 1);
            auto bulk = client->Expose(url, message.c_str(), message.length(), 0);
            auto event = client->Send(bulk);
            send_events.push_back(std::move(event));
        }
        
        // Post multiple receives
        for (int i = 0; i < num_messages; ++i) {
            auto buffer = std::make_unique<char[]>(1024);
            auto event = client->Recv(buffer.get(), 1024, url);
            recv_events.push_back(std::move(event));
            recv_buffers.push_back(std::move(buffer));
        }
        
        bool all_sends_queued = true;
        bool all_recvs_queued = true;
        
        for (const auto& event : send_events) {
            if (!event) {
                all_sends_queued = false;
                break;
            }
        }
        
        for (const auto& event : recv_events) {
            if (!event) {
                all_recvs_queued = false;
                break;
            }
        }
        
        log_result("Multiple sends queued successfully", all_sends_queued,
                  "Queued " + std::to_string(send_events.size()) + " send operations");
        
        log_result("Multiple recvs queued successfully", all_recvs_queued,
                  "Queued " + std::to_string(recv_events.size()) + " recv operations");
        
        // Cleanup
        client->Disconnect(url);
        server->Stop();
    }
    
    void test_process_completions_queue() {
        std::cout << "\n=== Testing ProcessCompletions Queue Logic ===" << std::endl;
        
        auto server = hshm::lbm::Transport::CreateServer(hshm::lbm::TransportType::TCP);
        auto client = hshm::lbm::Transport::CreateClient(hshm::lbm::TransportType::TCP);
        
        std::string url = "tcp://127.0.0.1:5557";
        server->StartServer(url, hshm::lbm::TransportType::TCP);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        
        client->Connect(url, hshm::lbm::TransportType::TCP);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        
        // Send a message
        std::string message = "ProcessCompletions test";
        auto bulk = client->Expose(url, message.c_str(), message.length(), 0);
        auto send_event = client->Send(bulk);
        
        char recv_buffer[1024];
        auto recv_event = client->Recv(recv_buffer, sizeof(recv_buffer), url);
        
        // Test that ProcessCompletions actually processes the queue
        bool initial_send_done = send_event->is_done;
        bool initial_recv_done = recv_event->is_done;
        
        // Process messages on server side
        for (int i = 0; i < 10; ++i) {
            server->ProcessMessages();
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        
        // Process completions on client side
        for (int i = 0; i < 10; ++i) {
            client->ProcessCompletions(10.0);
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        
        bool final_send_done = send_event->is_done;
        bool final_recv_done = recv_event->is_done;
        
        bool process_completions_works = final_send_done && final_recv_done;
        
        log_result("ProcessCompletions processes queue", process_completions_works,
                  "Initial: send=" + std::to_string(initial_send_done) + 
                  ", recv=" + std::to_string(initial_recv_done) +
                  " Final: send=" + std::to_string(final_send_done) + 
                  ", recv=" + std::to_string(final_recv_done));
        
        // Cleanup
        client->Disconnect(url);
        server->Stop();
    }
    
    void test_event_timeouts() {
        std::cout << "\n=== Testing Event Timeouts ===" << std::endl;
        
        auto client = hshm::lbm::Transport::CreateClient(hshm::lbm::TransportType::TCP);
        
        // Try to connect to non-existent server
        std::string url = "tcp://127.0.0.1:9999";
        client->Connect(url, hshm::lbm::TransportType::TCP);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        
        // Send message to non-existent server
        std::string message = "Timeout test";
        auto bulk = client->Expose(url, message.c_str(), message.length(), 0);
        auto send_event = client->Send(bulk);
        
        // Check that event has timer
        bool has_start_time = (send_event->start_time != std::chrono::steady_clock::time_point{});
        log_result("Event has start time", has_start_time);
        
        // Wait and check timeout
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        
        // Process completions should handle timeout
        client->ProcessCompletions(50.0); // 50ms timeout
        
        bool timeout_handled = send_event->is_done && (send_event->error_code != 0);
        log_result("Event timeout handled", timeout_handled,
                  "is_done=" + std::to_string(send_event->is_done) + 
                  ", error_code=" + std::to_string(send_event->error_code));
        
        // Cleanup
        client->Disconnect(url);
    }
    
    void test_binary_data_serialization() {
        std::cout << "\n=== Testing Binary Data Serialization ===" << std::endl;
        
        auto server = hshm::lbm::Transport::CreateServer(hshm::lbm::TransportType::TCP);
        auto client = hshm::lbm::Transport::CreateClient(hshm::lbm::TransportType::TCP);
        
        std::string url = "tcp://127.0.0.1:5558";
        server->StartServer(url, hshm::lbm::TransportType::TCP);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        
        client->Connect(url, hshm::lbm::TransportType::TCP);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        
        // Create binary data with null bytes
        const size_t data_size = 100;
        auto binary_data = std::make_unique<uint8_t[]>(data_size);
        
        // Fill with random binary data including null bytes
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, 255);
        
        for (size_t i = 0; i < data_size; ++i) {
            binary_data[i] = static_cast<uint8_t>(dis(gen));
        }
        
        // Ensure we have some null bytes
        binary_data[10] = 0;
        binary_data[50] = 0;
        binary_data[90] = 0;
        
        auto bulk = client->Expose(url, reinterpret_cast<char*>(binary_data.get()), data_size, 0);
        auto send_event = client->Send(bulk);
        
        auto recv_buffer = std::make_unique<char[]>(data_size + 100);
        auto recv_event = client->Recv(recv_buffer.get(), data_size + 100, url);
        
        // Process the exchange
        for (int i = 0; i < 20; ++i) {
            server->ProcessMessages();
            client->ProcessCompletions(10.0);
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            
            if (send_event->is_done && recv_event->is_done) {
                break;
            }
        }
        
        bool exchange_completed = send_event->is_done && recv_event->is_done &&
                                 send_event->error_code == 0 && recv_event->error_code == 0;
        
        log_result("Binary data exchange completed", exchange_completed,
                  "Send done=" + std::to_string(send_event->is_done) + 
                  ", Recv done=" + std::to_string(recv_event->is_done) +
                  ", Send error=" + std::to_string(send_event->error_code) +
                  ", Recv error=" + std::to_string(recv_event->error_code));
        
        // Verify data integrity
        bool data_integrity = false;
        if (exchange_completed && recv_event->bytes_transferred == data_size) {
            data_integrity = true;
            for (size_t i = 0; i < data_size; ++i) {
                if (static_cast<uint8_t>(recv_buffer[i]) != binary_data[i]) {
                    data_integrity = false;
                    break;
                }
            }
        }
        
        log_result("Binary data integrity preserved", data_integrity,
                  "Expected size=" + std::to_string(data_size) + 
                  ", Received size=" + std::to_string(recv_event->bytes_transferred));
        
        // Cleanup
        client->Disconnect(url);
        server->Stop();
    }
    
    void run_all_tests() {
        std::cout << "LightBeam Non-blocking Requirements Test Suite" << std::endl;
        std::cout << "===============================================" << std::endl;
        
        test_nonblocking_send_recv();
        test_multiple_concurrent_messages();
        test_process_completions_queue();
        test_event_timeouts();
        test_binary_data_serialization();
        
        // Print summary
        std::cout << "\n=== Test Summary ===" << std::endl;
        int passed = 0;
        int total = 0;
        
        for (const auto& result : test_results) {
            std::cout << result << std::endl;
            total++;
            if (result.find("[PASS]") != std::string::npos) {
                passed++;
            }
        }
        
        std::cout << "\nPassed: " << passed << "/" << total << " tests" << std::endl;
        
        if (passed == total) {
            std::cout << "🎉 All requirements satisfied!" << std::endl;
        } else {
            std::cout << "❌ Some requirements need attention." << std::endl;
        }
    }
};

int main() {
    NonBlockingTest test;
    test.run_all_tests();
    return 0;
} 