#include "hermes_shm/lightbeam/lightbeam.h"
#include "hermes_shm/util/timer.h"
#include <iostream>
#include <thread>
#include <chrono>

class SimpleClientTest {
public:
    void test_nonblocking_send() {
        std::cout << "\n=== Test 1: Non-blocking Send ===" << std::endl;
        
        auto client = hshm::lbm::Transport::CreateClient(hshm::lbm::TransportType::TCP);
        std::string url = "tcp://127.0.0.1:5555";
        
        // Wait for server
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        client->Connect(url, hshm::lbm::TransportType::TCP);
        
        // Test: Send should return immediately
        std::string message = "Hello Server!";
        auto bulk = client->Expose(url, message.c_str(), message.length(), 0);
        
        hshm::Timer timer;
        timer.Resume();
        auto event = client->Send(bulk);  // This should return immediately
        timer.Pause();
        
        double send_time = timer.GetMsecFromStart();
        
        std::cout << "Send time: " << send_time << " ms" << std::endl;
        std::cout << "Event ID: " << event->event_id << std::endl;
        std::cout << "Initially done: " << (event->is_done ? "YES" : "NO") << std::endl;
        
        if (send_time < 10.0) {  // Should be very fast (< 10ms)
            std::cout << "✅ PASS: Send is non-blocking" << std::endl;
        } else {
            std::cout << "❌ FAIL: Send took too long" << std::endl;
        }
        
        client->Disconnect(url);
    }
    
    void test_round_trip_timing() {
        std::cout << "\n=== Test 2: Round-trip Timing ===" << std::endl;
        
        auto client = hshm::lbm::Transport::CreateClient(hshm::lbm::TransportType::TCP);
        std::string url = "tcp://127.0.0.1:5555";
        
        // Wait for server
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        client->Connect(url, hshm::lbm::TransportType::TCP);
        
        // Test round-trip as requested
        std::string message = "Round-trip test";
        auto bulk = client->Expose(url, message.c_str(), message.length(), 0);
        char recv_buffer[1024];
        
        hshm::Timer timer;
        timer.Resume();
        
        // Send message
        auto send_event = client->Send(bulk);
        auto recv_event = client->Recv(recv_buffer, sizeof(recv_buffer), url);
        
        // Wait for completion with timeout
        bool completed = false;
        auto start = std::chrono::steady_clock::now();
        
        while (!completed) {
            client->ProcessCompletions(100.0);
            
            if (send_event->is_done && recv_event->is_done) {
                completed = true;
                timer.Pause();
                break;
            }
            
            // 3-second timeout
            auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            if (elapsed > 3.0) {
                timer.Pause();
                break;
            }
            
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        
        double round_trip_time = timer.GetMsecFromStart();
        
        std::cout << "Round-trip time: " << round_trip_time << " ms" << std::endl;
        std::cout << "Send completed: " << (send_event->is_done ? "YES" : "NO") << std::endl;
        std::cout << "Recv completed: " << (recv_event->is_done ? "YES" : "NO") << std::endl;
        std::cout << "Send bytes: " << send_event->bytes_transferred << std::endl;
        std::cout << "Recv bytes: " << recv_event->bytes_transferred << std::endl;
        
        if (completed && round_trip_time < 1000.0) {
            std::cout << "✅ PASS: Round-trip successful" << std::endl;
        } else {
            std::cout << "❌ FAIL: Round-trip failed or too slow" << std::endl;
        }
        
        client->Disconnect(url);
    }
    
    void test_process_completions() {
        std::cout << "\n=== Test 3: ProcessCompletions Function ===" << std::endl;
        
        auto client = hshm::lbm::Transport::CreateClient(hshm::lbm::TransportType::TCP);
        std::string url = "tcp://127.0.0.1:5555";
        
        // Wait for server
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        client->Connect(url, hshm::lbm::TransportType::TCP);
        
        // Send a message
        std::string message = "ProcessCompletions test";
        auto bulk = client->Expose(url, message.c_str(), message.length(), 0);
        auto event = client->Send(bulk);
        
        std::cout << "Before ProcessCompletions: " << (event->is_done ? "DONE" : "PENDING") << std::endl;
        
        // Call ProcessCompletions to handle completion
        for (int i = 0; i < 10; ++i) {
            client->ProcessCompletions(100.0);
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        
        std::cout << "After ProcessCompletions: " << (event->is_done ? "DONE" : "PENDING") << std::endl;
        
        if (event->is_done) {
            std::cout << "✅ PASS: ProcessCompletions works" << std::endl;
        } else {
            std::cout << "❌ FAIL: ProcessCompletions didn't complete operation" << std::endl;
        }
        
        client->Disconnect(url);
    }
    
    void test_concurrent_messaging() {
        std::cout << "\n=== Test 4: Concurrent Messaging ===" << std::endl;
        
        auto client = hshm::lbm::Transport::CreateClient(hshm::lbm::TransportType::TCP);
        std::string url = "tcp://127.0.0.1:5555";
        
        std::cout << "Connecting to server..." << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        client->Connect(url, hshm::lbm::TransportType::TCP);
        
        const int NUM_CONCURRENT_MESSAGES = 10;
        std::vector<std::string> messages;
        std::vector<std::unique_ptr<hshm::lbm::Event>> sent_events;
        std::vector<std::unique_ptr<hshm::lbm::Event>> recv_events;
        std::vector<hshm::lbm::Bulk> bulks;
        std::vector<std::unique_ptr<char[]>> recv_buffers;
        
        // Prepare messages and bulks
        for (int i = 0; i < NUM_CONCURRENT_MESSAGES; i++) {
            messages.push_back("Concurrent message " + std::to_string(i));
            auto bulk = client->Expose(url, messages[i].c_str(), messages[i].length(), 0);
            bulks.push_back(bulk);
            
            // Prepare receive buffer
            recv_buffers.push_back(std::make_unique<char[]>(1024));
        }
        
        hshm::Timer timer;
        timer.Resume();
        
        // Send all messages concurrently (non-blocking)
        std::cout << "Sending " << NUM_CONCURRENT_MESSAGES << " messages concurrently..." << std::endl;
        for (int i = 0; i < NUM_CONCURRENT_MESSAGES; i++) {
            auto send_event = client->Send(bulks[i]);
            sent_events.push_back(std::move(send_event));
        }
        
        // Start receiving all messages concurrently
        std::cout << "Starting concurrent receives..." << std::endl;
        for (int i = 0; i < NUM_CONCURRENT_MESSAGES; i++) {
            auto recv_event = client->Recv(recv_buffers[i].get(), 1024, url);
            recv_events.push_back(std::move(recv_event));
        }
        
        // Process completions for all operations
        std::cout << "Processing completions..." << std::endl;
        int completed_sends = 0;
        int completed_receives = 0;
        int max_iterations = 1000;
        int iterations = 0;
        
        while ((completed_sends < NUM_CONCURRENT_MESSAGES || completed_receives < NUM_CONCURRENT_MESSAGES) 
               && iterations < max_iterations) {
            
            client->ProcessCompletions(100.0);
            
            // Check send completions
            completed_sends = 0;
            for (auto& event : sent_events) {
                if (event->is_done && event->error_code == 0) {
                    completed_sends++;
                }
            }
            
            // Check receive completions
            completed_receives = 0;
            for (auto& event : recv_events) {
                if (event->is_done && event->error_code == 0) {
                    completed_receives++;
                }
            }
            
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            iterations++;
        }
        
        timer.Pause();
        double test_time = timer.GetMsecFromStart();
        
        std::cout << "Concurrent messaging test results:" << std::endl;
        std::cout << "  Test duration: " << test_time << " ms" << std::endl;
        std::cout << "  Completed sends: " << completed_sends << "/" << NUM_CONCURRENT_MESSAGES << std::endl;
        std::cout << "  Completed receives: " << completed_receives << "/" << NUM_CONCURRENT_MESSAGES << std::endl;
        std::cout << "  Processing iterations: " << iterations << std::endl;
        
        bool sends_ok = (completed_sends == NUM_CONCURRENT_MESSAGES);
        bool receives_ok = (completed_receives == NUM_CONCURRENT_MESSAGES);
        bool performance_ok = (test_time < 5000.0); // Should complete within 5 seconds
        
        if (sends_ok && receives_ok && performance_ok) {
            std::cout << "✅ PASS: Concurrent messaging works correctly" << std::endl;
        } else {
            std::cout << "❌ FAIL: Concurrent messaging issues detected" << std::endl;
            if (!sends_ok) std::cout << "  - Send operations incomplete" << std::endl;
            if (!receives_ok) std::cout << "  - Receive operations incomplete" << std::endl;
            if (!performance_ok) std::cout << "  - Performance too slow" << std::endl;
        }
        
        client->Disconnect(url);
    }

    void test_mixed_concurrent_operations() {
        std::cout << "\n=== Test 5: Mixed Concurrent Operations ===" << std::endl;
        
        auto client = hshm::lbm::Transport::CreateClient(hshm::lbm::TransportType::TCP);
        std::string url = "tcp://127.0.0.1:5555";
        
        std::cout << "Connecting to server..." << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        client->Connect(url, hshm::lbm::TransportType::TCP);
        
        const int NUM_OPERATIONS = 20;
        std::vector<std::unique_ptr<hshm::lbm::Event>> operations;
        std::vector<hshm::lbm::Bulk> bulks;
        std::vector<std::unique_ptr<char[]>> recv_buffers;
        int send_count = 0;
        int recv_count = 0;
        
        hshm::Timer timer;
        timer.Resume();
        
        // Mix sends and receives
        std::cout << "Starting mixed concurrent operations..." << std::endl;
        for (int i = 0; i < NUM_OPERATIONS; i++) {
            if (i % 2 == 0) {
                // Send operation
                std::string message = "Mixed send " + std::to_string(i);
                auto bulk = client->Expose(url, message.c_str(), message.length(), 0);
                bulks.push_back(bulk);
                
                auto send_event = client->Send(bulks.back());
                operations.push_back(std::move(send_event));
                send_count++;
            } else {
                // Receive operation
                auto recv_buffer = std::make_unique<char[]>(1024);
                auto recv_event = client->Recv(recv_buffer.get(), 1024, url);
                recv_buffers.push_back(std::move(recv_buffer));
                operations.push_back(std::move(recv_event));
                recv_count++;
            }
        }
        
        // Process all completions
        std::cout << "Processing mixed completions..." << std::endl;
        int completed_ops = 0;
        int max_iterations = 1000;
        int iterations = 0;
        
        while (completed_ops < NUM_OPERATIONS && iterations < max_iterations) {
            client->ProcessCompletions(100.0);
            
            completed_ops = 0;
            for (auto& event : operations) {
                if (event->is_done && event->error_code == 0) {
                    completed_ops++;
                }
            }
            
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            iterations++;
        }
        
        timer.Pause();
        double test_time = timer.GetMsecFromStart();
        
        std::cout << "Mixed operations test results:" << std::endl;
        std::cout << "  Test duration: " << test_time << " ms" << std::endl;
        std::cout << "  Send operations: " << send_count << std::endl;
        std::cout << "  Receive operations: " << recv_count << std::endl;
        std::cout << "  Completed operations: " << completed_ops << "/" << NUM_OPERATIONS << std::endl;
        std::cout << "  Processing iterations: " << iterations << std::endl;
        
        bool completion_ok = (completed_ops == NUM_OPERATIONS);
        bool performance_ok = (test_time < 5000.0);
        
        if (completion_ok && performance_ok) {
            std::cout << "✅ PASS: Mixed concurrent operations work correctly" << std::endl;
        } else {
            std::cout << "❌ FAIL: Mixed concurrent operations issues detected" << std::endl;
            if (!completion_ok) std::cout << "  - Operations incomplete" << std::endl;
            if (!performance_ok) std::cout << "  - Performance too slow" << std::endl;
        }
        
        client->Disconnect(url);
    }

    void run_all_tests() {
        std::cout << "Simple LightBeam Client Unit Tests" << std::endl;
        std::cout << "===================================" << std::endl;
        
        test_nonblocking_send();
        test_round_trip_timing();
        test_process_completions();
        test_concurrent_messaging();
        test_mixed_concurrent_operations();
        
        std::cout << "\n=== Test Summary ===" << std::endl;
        std::cout << "Tests completed. Check output above for PASS/FAIL status." << std::endl;
    }
};

int main() {
    SimpleClientTest test;
    test.run_all_tests();
    return 0;
} 