# HSHM Atomic Types Guide

## Overview

The Atomic Types API in Hermes Shared Memory (HSHM) provides cross-platform atomic operations with support for CPU, GPU (CUDA/ROCm), and non-atomic variants. The API abstracts platform differences and provides consistent atomic operations for thread-safe programming across different execution environments.

## Atomic Type Variants

### Platform-Specific Atomic Types

```cpp
#include "hermes_shm/types/atomic.h"

void atomic_variants_example() {
    // Standard atomic (uses std::atomic on host, GPU atomics on device)
    hshm::ipc::atomic<int> standard_atomic(42);
    
    // Non-atomic (for single-threaded or externally synchronized code)
    hshm::ipc::nonatomic<int> non_atomic_value(100);
    
    // Explicit GPU atomic (CUDA/ROCm specific)
#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM
    hshm::ipc::rocm_atomic<int> gpu_atomic(200);
#endif
    
    // Explicit standard library atomic
    hshm::ipc::std_atomic<int> std_lib_atomic(300);
    
    // Conditional atomic - chooses atomic or non-atomic based on template parameter
    hshm::ipc::opt_atomic<int, true>  conditional_atomic(400);     // Uses atomic
    hshm::ipc::opt_atomic<int, false> conditional_nonatomic(500); // Uses nonatomic
    
    printf("Standard atomic: %d\n", standard_atomic.load());
    printf("Non-atomic: %d\n", non_atomic_value.load());
    printf("Conditional atomic: %d\n", conditional_atomic.load());
}
```

## Basic Atomic Operations

### Load, Store, and Exchange

```cpp
void basic_atomic_operations() {
    hshm::ipc::atomic<int> counter(0);
    
    // Load value
    int current = counter.load();
    printf("Current value: %d\n", current);
    
    // Store new value
    counter.store(10);
    printf("After store(10): %d\n", counter.load());
    
    // Exchange (atomically set new value and return old)
    int old_value = counter.exchange(20);
    printf("Exchange returned: %d, new value: %d\n", old_value, counter.load());
    
    // Compare and exchange (conditional atomic update)
    int expected = 20;
    bool success = counter.compare_exchange_weak(expected, 30);
    printf("CAS success: %s, value: %d\n", success ? "yes" : "no", counter.load());
    
    // Try CAS with wrong expected value
    expected = 25;  // Wrong expected value
    success = counter.compare_exchange_strong(expected, 40);
    printf("CAS with wrong expected: %s, value: %d, expected now: %d\n", 
           success ? "yes" : "no", counter.load(), expected);
}
```

### Arithmetic Operations

```cpp
void arithmetic_operations_example() {
    hshm::ipc::atomic<int> counter(10);
    
    // Fetch and add
    int old_val = counter.fetch_add(5);
    printf("fetch_add(5): old=%d, new=%d\n", old_val, counter.load());
    
    // Fetch and subtract
    old_val = counter.fetch_sub(3);
    printf("fetch_sub(3): old=%d, new=%d\n", old_val, counter.load());
    
    // Increment operators
    ++counter;  // Pre-increment
    printf("After pre-increment: %d\n", counter.load());
    
    counter++;  // Post-increment
    printf("After post-increment: %d\n", counter.load());
    
    // Decrement operators
    --counter;  // Pre-decrement
    printf("After pre-decrement: %d\n", counter.load());
    
    counter--;  // Post-decrement
    printf("After post-decrement: %d\n", counter.load());
    
    // Assignment operators
    counter += 10;
    printf("After += 10: %d\n", counter.load());
    
    counter -= 5;
    printf("After -= 5: %d\n", counter.load());
}
```

### Bitwise Operations

```cpp
void bitwise_operations_example() {
    hshm::ipc::atomic<uint32_t> flags(0xF0F0F0F0);
    
    printf("Initial flags: 0x%08X\n", flags.load());
    
    // Bitwise AND
    uint32_t result = (flags & 0xFF00FF00).load();
    printf("flags & 0xFF00FF00 = 0x%08X\n", result);
    
    // Bitwise OR
    result = (flags | 0x0F0F0F0F).load();
    printf("flags | 0x0F0F0F0F = 0x%08X\n", result);
    
    // Bitwise XOR
    result = (flags ^ 0xFFFFFFFF).load();
    printf("flags ^ 0xFFFFFFFF = 0x%08X\n", result);
    
    // Assignment bitwise operations
    flags &= 0xFF00FF00;
    printf("After &= 0xFF00FF00: 0x%08X\n", flags.load());
    
    flags |= 0x0F0F0F0F;
    printf("After |= 0x0F0F0F0F: 0x%08X\n", flags.load());
    
    flags ^= 0x12345678;
    printf("After ^= 0x12345678: 0x%08X\n", flags.load());
}
```

## Thread-Safe Programming Patterns

### Reference Counting

```cpp
class RefCountedResource {
    mutable hshm::ipc::atomic<int> ref_count_;
    std::string resource_name_;
    
public:
    explicit RefCountedResource(const std::string& name) 
        : ref_count_(1), resource_name_(name) {
        printf("Resource '%s' created with ref count 1\n", name.c_str());
    }
    
    ~RefCountedResource() {
        printf("Resource '%s' destroyed\n", resource_name_.c_str());
    }
    
    void AddRef() const {
        int old_count = ref_count_.fetch_add(1);
        printf("AddRef: %s ref count %d -> %d\n", 
               resource_name_.c_str(), old_count, old_count + 1);
    }
    
    void Release() const {
        int old_count = ref_count_.fetch_sub(1);
        printf("Release: %s ref count %d -> %d\n", 
               resource_name_.c_str(), old_count, old_count - 1);
        
        if (old_count == 1) {
            delete this;  // Last reference released
        }
    }
    
    int GetRefCount() const {
        return ref_count_.load();
    }
    
    const std::string& GetName() const {
        return resource_name_;
    }
};

// Smart pointer for reference-counted resources
template<typename T>
class atomic_shared_ptr {
    T* ptr_;
    
public:
    explicit atomic_shared_ptr(T* ptr = nullptr) : ptr_(ptr) {
        if (ptr_) ptr_->AddRef();
    }
    
    atomic_shared_ptr(const atomic_shared_ptr& other) : ptr_(other.ptr_) {
        if (ptr_) ptr_->AddRef();
    }
    
    atomic_shared_ptr& operator=(const atomic_shared_ptr& other) {
        if (this != &other) {
            if (ptr_) ptr_->Release();
            ptr_ = other.ptr_;
            if (ptr_) ptr_->AddRef();
        }
        return *this;
    }
    
    ~atomic_shared_ptr() {
        if (ptr_) ptr_->Release();
    }
    
    T* operator->() const { return ptr_; }
    T& operator*() const { return *ptr_; }
    T* get() const { return ptr_; }
    
    bool operator==(const atomic_shared_ptr& other) const {
        return ptr_ == other.ptr_;
    }
};

void ref_counting_example() {
    // Create resource
    auto resource = atomic_shared_ptr<RefCountedResource>(
        new RefCountedResource("TestResource"));
    
    printf("Initial ref count: %d\n", resource->GetRefCount());
    
    // Create additional references
    {
        auto resource2 = resource;  // Copy constructor
        auto resource3 = resource;  // Another copy
        
        printf("With 3 references: %d\n", resource->GetRefCount());
        
        // resource2 and resource3 go out of scope here
    }
    
    printf("After scope exit: %d\n", resource->GetRefCount());
    
    // resource goes out of scope and destroys the object
}
```

### Lock-Free Data Structures

```cpp
template<typename T>
class LockFreeStack {
    struct Node {
        T data;
        Node* next;
        
        Node(T&& data) : data(std::move(data)), next(nullptr) {}
    };
    
    hshm::ipc::atomic<Node*> head_;
    
public:
    LockFreeStack() : head_(nullptr) {}
    
    ~LockFreeStack() {
        while (Node* old_head = head_.load()) {
            head_ = old_head->next;
            delete old_head;
        }
    }
    
    void Push(T data) {
        Node* new_node = new Node(std::move(data));
        
        Node* old_head = head_.load();
        do {
            new_node->next = old_head;
        } while (!head_.compare_exchange_weak(old_head, new_node));
    }
    
    bool Pop(T& result) {
        Node* old_head = head_.load();
        
        while (old_head && 
               !head_.compare_exchange_weak(old_head, old_head->next)) {
            // Loop until successful CAS or stack is empty
        }
        
        if (old_head) {
            result = std::move(old_head->data);
            delete old_head;
            return true;
        }
        
        return false;  // Stack was empty
    }
    
    bool IsEmpty() const {
        return head_.load() == nullptr;
    }
};

void lock_free_stack_example() {
    LockFreeStack<int> stack;
    std::vector<std::thread> producers;
    std::vector<std::thread> consumers;
    
    const int num_producers = 2;
    const int num_consumers = 2;
    const int items_per_producer = 100;
    
    // Start producer threads
    for (int i = 0; i < num_producers; ++i) {
        producers.emplace_back([&stack, i, items_per_producer]() {
            for (int j = 0; j < items_per_producer; ++j) {
                int value = i * items_per_producer + j;
                stack.Push(value);
                printf("Producer %d pushed %d\n", i, value);
            }
        });
    }
    
    // Start consumer threads
    hshm::ipc::atomic<int> total_consumed(0);
    for (int i = 0; i < num_consumers; ++i) {
        consumers.emplace_back([&stack, &total_consumed, i]() {
            int value;
            int consumed = 0;
            
            while (total_consumed.load() < num_producers * items_per_producer) {
                if (stack.Pop(value)) {
                    printf("Consumer %d popped %d\n", i, value);
                    consumed++;
                    total_consumed.fetch_add(1);
                } else {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
            }
            
            printf("Consumer %d finished, consumed %d items\n", i, consumed);
        });
    }
    
    // Wait for completion
    for (auto& producer : producers) {
        producer.join();
    }
    
    for (auto& consumer : consumers) {
        consumer.join();
    }
    
    printf("Final total consumed: %d\n", total_consumed.load());
    printf("Stack empty: %s\n", stack.IsEmpty() ? "yes" : "no");
}
```

### Atomic Flags and Signaling

```cpp
class WorkerCoordinator {
    hshm::ipc::atomic<bool> start_flag_;
    hshm::ipc::atomic<bool> stop_flag_;
    hshm::ipc::atomic<int> ready_workers_;
    hshm::ipc::atomic<int> completed_workers_;
    const int total_workers_;
    
public:
    explicit WorkerCoordinator(int num_workers) 
        : start_flag_(false), stop_flag_(false), 
          ready_workers_(0), completed_workers_(0),
          total_workers_(num_workers) {}
    
    void WorkerReady() {
        int ready = ready_workers_.fetch_add(1) + 1;
        printf("Worker ready (%d/%d)\n", ready, total_workers_);
        
        // If all workers are ready, signal start
        if (ready == total_workers_) {
            start_flag_.store(true);
            printf("All workers ready, starting work!\n");
        }
    }
    
    void WaitForStart() {
        // Busy wait for start signal
        while (!start_flag_.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    
    bool ShouldStop() {
        return stop_flag_.load();
    }
    
    void WorkerCompleted() {
        int completed = completed_workers_.fetch_add(1) + 1;
        printf("Worker completed (%d/%d)\n", completed, total_workers_);
    }
    
    void SignalStop() {
        stop_flag_.store(true);
        printf("Stop signal sent to all workers\n");
    }
    
    void WaitForCompletion() {
        while (completed_workers_.load() < total_workers_) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        printf("All workers completed\n");
    }
    
    void Reset() {
        start_flag_ = false;
        stop_flag_ = false;
        ready_workers_ = 0;
        completed_workers_ = 0;
    }
};

void worker_coordination_example() {
    const int num_workers = 4;
    WorkerCoordinator coordinator(num_workers);
    std::vector<std::thread> workers;
    
    // Launch worker threads
    for (int i = 0; i < num_workers; ++i) {
        workers.emplace_back([&coordinator, i]() {
            printf("Worker %d initializing...\n", i);
            
            // Simulate initialization time
            std::this_thread::sleep_for(std::chrono::milliseconds(100 * (i + 1)));
            
            // Signal ready and wait for start
            coordinator.WorkerReady();
            coordinator.WaitForStart();
            
            // Do work until stop signal
            int work_count = 0;
            while (!coordinator.ShouldStop()) {
                // Simulate work
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
                work_count++;
                
                printf("Worker %d completed work unit %d\n", i, work_count);
                
                // Stop after some work for demo purposes
                if (work_count >= 3) break;
            }
            
            coordinator.WorkerCompleted();
            printf("Worker %d finished after %d work units\n", i, work_count);
        });
    }
    
    // Let workers run for a while
    std::this_thread::sleep_for(std::chrono::seconds(2));
    
    // Signal stop and wait for completion
    coordinator.SignalStop();
    coordinator.WaitForCompletion();
    
    // Join all threads
    for (auto& worker : workers) {
        worker.join();
    }
    
    printf("All workers joined\n");
}
```

## Cross-Platform GPU Atomics

```cpp
#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM

__global__ void gpu_atomic_kernel(hshm::ipc::atomic<int>* counter, int num_threads) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < num_threads) {
        // Atomic increment on GPU
        int old_val = counter->fetch_add(1);
        
        // GPU-specific atomic operations
        counter->fetch_sub(1);  // Decrement
        counter->fetch_add(2);  // Add 2
        
        // Compare and swap
        int expected = old_val;
        counter->compare_exchange_weak(expected, old_val + 10);
    }
}

void gpu_atomic_example() {
    // Allocate unified memory for atomic variable
    hshm::ipc::atomic<int>* d_counter;
    cudaMallocManaged(&d_counter, sizeof(hshm::ipc::atomic<int>));
    
    // Initialize counter
    new(d_counter) hshm::ipc::atomic<int>(0);
    
    const int num_threads = 1024;
    const int block_size = 256;
    const int num_blocks = (num_threads + block_size - 1) / block_size;
    
    printf("Initial counter value: %d\n", d_counter->load());
    
    // Launch GPU kernel
    gpu_atomic_kernel<<<num_blocks, block_size>>>(d_counter, num_threads);
    cudaDeviceSynchronize();
    
    printf("Final counter value after GPU kernel: %d\n", d_counter->load());
    
    // Cleanup
    d_counter->~atomic();
    cudaFree(d_counter);
}

#endif
```

## Memory Ordering and Synchronization

```cpp
void memory_ordering_example() {
    hshm::ipc::atomic<int> data(0);
    hshm::ipc::atomic<bool> flag(false);
    
    std::thread producer([&]() {
        // Write data
        data.store(42, std::memory_order_relaxed);
        
        // Signal data is ready
        flag.store(true, std::memory_order_release);  // Release semantics
        
        printf("Producer: data written and flag set\n");
    });
    
    std::thread consumer([&]() {
        // Wait for flag
        while (!flag.load(std::memory_order_acquire)) {  // Acquire semantics
            std::this_thread::sleep_for(std::chrono::microseconds(1));
        }
        
        // Read data (guaranteed to see the write due to acquire-release)
        int value = data.load(std::memory_order_relaxed);
        printf("Consumer: read data value %d\n", value);
    });
    
    producer.join();
    consumer.join();
}
```

## Non-Atomic Variants for Performance

```cpp
void nonatomic_performance_comparison() {
    const int iterations = 1000000;
    
    // Timing atomic operations
    hshm::ipc::atomic<int> atomic_counter(0);
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; ++i) {
        atomic_counter.fetch_add(1);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto atomic_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    
    // Timing non-atomic operations
    hshm::ipc::nonatomic<int> nonatomic_counter(0);
    start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; ++i) {
        nonatomic_counter.fetch_add(1);
    }
    
    end = std::chrono::high_resolution_clock::now();
    auto nonatomic_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    
    printf("Performance comparison (%d iterations):\n", iterations);
    printf("Atomic operations:     %ld ns (%.2f ns/op)\n", 
           atomic_time.count(), double(atomic_time.count()) / iterations);
    printf("Non-atomic operations: %ld ns (%.2f ns/op)\n", 
           nonatomic_time.count(), double(nonatomic_time.count()) / iterations);
    printf("Speedup: %.2fx\n", 
           double(atomic_time.count()) / double(nonatomic_time.count()));
    
    printf("Final values - atomic: %d, non-atomic: %d\n",
           atomic_counter.load(), nonatomic_counter.load());
}
```

## Conditional Atomic Types

```cpp
template<bool THREAD_SAFE>
class ConfigurableCounter {
    hshm::ipc::opt_atomic<int, THREAD_SAFE> count_;
    
public:
    ConfigurableCounter() : count_(0) {}
    
    void Increment() {
        count_.fetch_add(1);
    }
    
    void Add(int value) {
        count_.fetch_add(value);
    }
    
    int Get() const {
        return count_.load();
    }
    
    void Reset() {
        count_.store(0);
    }
};

void conditional_atomic_example() {
    // Thread-safe version
    ConfigurableCounter<true> thread_safe_counter;
    
    // Non-atomic version for single-threaded use
    ConfigurableCounter<false> fast_counter;
    
    const int iterations = 100000;
    
    // Test thread-safe version with multiple threads
    std::vector<std::thread> threads;
    for (int i = 0; i < 4; ++i) {
        threads.emplace_back([&thread_safe_counter, iterations]() {
            for (int j = 0; j < iterations; ++j) {
                thread_safe_counter.Increment();
            }
        });
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    // Test non-atomic version (single-threaded)
    for (int i = 0; i < 4 * iterations; ++i) {
        fast_counter.Increment();
    }
    
    printf("Thread-safe counter: %d\n", thread_safe_counter.Get());
    printf("Fast counter: %d\n", fast_counter.Get());
    printf("Both should equal: %d\n", 4 * iterations);
}
```

## Serialization Support

```cpp
#include <sstream>
#include <cereal/archives/binary.hpp>

void atomic_serialization_example() {
    hshm::ipc::atomic<int> counter(12345);
    hshm::ipc::nonatomic<double> value(3.14159);
    
    // Serialize to binary stream
    std::stringstream ss;
    {
        cereal::BinaryOutputArchive archive(ss);
        archive(counter, value);
    }
    
    // Deserialize from binary stream
    hshm::ipc::atomic<int> loaded_counter;
    hshm::ipc::nonatomic<double> loaded_value;
    {
        cereal::BinaryInputArchive archive(ss);
        archive(loaded_counter, loaded_value);
    }
    
    printf("Original counter: %d, loaded: %d\n", 
           counter.load(), loaded_counter.load());
    printf("Original value: %f, loaded: %f\n", 
           value.load(), loaded_value.load());
}
```

## Best Practices

1. **Platform Selection**: Use `hshm::ipc::atomic<T>` for automatic platform selection (CPU vs GPU)
2. **Performance**: Use `nonatomic<T>` for single-threaded code or when external synchronization is provided
3. **Memory Ordering**: Specify appropriate memory ordering for performance-critical code
4. **GPU Compatibility**: Use HSHM atomic types for code that runs on both CPU and GPU
5. **Lock-Free Design**: Prefer atomic operations over locks for high-performance concurrent code
6. **Reference Counting**: Use atomic counters for thread-safe reference counting implementations
7. **Conditional Compilation**: Use `opt_atomic<T, bool>` for compile-time atomic vs non-atomic selection
8. **Cross-Platform**: All atomic types work consistently across different architectures and GPUs
9. **Serialization**: Atomic types support standard serialization for persistence and communication
10. **Testing**: Always test atomic code under high contention to verify correctness and performance