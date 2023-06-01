/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
* Distributed under BSD 3-Clause license.                                   *
* Copyright by The HDF Group.                                               *
* Copyright by the Illinois Institute of Technology.                        *
* All rights reserved.                                                      *
*                                                                           *
* This file is part of Hermes. The full Hermes copyright notice, including  *
* terms governing use, modification, and redistribution, is contained in    *
* the COPYING file, which can be found at the top directory. If you do not  *
* have access to the file, you may request a copy from help@hdfgroup.org.   *
* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include "basic_test.h"
#include "test_init.h"
#include "hermes_shm/data_structures/ipc/mpmc_queue.h"

template<typename QueueT, typename T>
class QueueTestSuite {
 public:
  hipc::uptr<QueueT> &queue_;

 public:
  /** Constructor */
  explicit QueueTestSuite(hipc::uptr<QueueT> &queue)
  : queue_(queue) {}

  /** Producer method */
  void Produce(size_t count_per_rank) {
    try {
      int rank = omp_get_thread_num();
      for (size_t i = 0; i < count_per_rank; ++i) {
        size_t idx = rank * count_per_rank + i;
        CREATE_SET_VAR_TO_INT_OR_STRING(T, var, idx);
        while (queue_->emplace(var).IsNull()) {}
      }
    } catch (hshm::Error &e) {
      HELOG(kFatal, e.what())
    }
  }

  /** Consumer method */
  void Consume(std::atomic<size_t> &count, size_t total_count) {
    std::vector<int> entries;
    entries.reserve(total_count);
    auto entry = hipc::make_uptr<T>();
    auto &entry_ref = *entry;

    // Consume everything
    while (count != total_count) {
      auto qtok = queue_->pop(entry_ref);
      if (qtok.IsNull()) {
        continue;
      }
      CREATE_GET_INT_FROM_VAR(T, entry_int, entry_ref)
      entries.emplace_back(entry_int);
      ++count;
    }

    // Ensure there's no data left in the queue
    REQUIRE(queue_->pop(entry_ref).IsNull());
  }
};

template<typename QueueT, typename T>
void ProduceThenConsume(size_t nproducers,
                        size_t nconsumers,
                        size_t count_per_rank) {
  size_t depth = 32;
  auto queue = hipc::make_uptr<QueueT>(depth);
  QueueTestSuite<QueueT, T> q(queue);
  std::atomic<size_t> count = 0;

  // Produce all the data
  omp_set_dynamic(0);
#pragma omp parallel shared(nproducers, count_per_rank, q, count) num_threads(nproducers)  // NOLINT
  {  // NOLINT
#pragma omp barrier
    q.Produce(count_per_rank);
#pragma omp barrier
  }

  omp_set_dynamic(0);
#pragma omp parallel shared(nproducers, count_per_rank, q) num_threads(nconsumers)  // NOLINT
  {  // NOLINT
#pragma omp barrier
    // Consume all the data
    q.Consume(count, count_per_rank * nproducers);
#pragma omp barrier
  }
}

template<typename QueueT, typename T>
void ProduceAndConsume(size_t nproducers,
                       size_t nconsumers,
                       size_t count_per_rank) {
  size_t depth = 32;
  auto queue = hipc::make_uptr<QueueT>(depth);
  size_t nthreads = nproducers + nconsumers;
  QueueTestSuite<QueueT, T> q(queue);
  std::atomic<size_t> count = 0;

  // Produce all the data
  omp_set_dynamic(0);
#pragma omp parallel shared(nproducers, count_per_rank, q, count) num_threads(nthreads)  // NOLINT
  {  // NOLINT
#pragma omp barrier
    size_t rank = omp_get_thread_num();
    if (rank < nproducers) {
      // Producer
      q.Produce(count_per_rank);
    } else {
      // Consumer
      q.Consume(count, count_per_rank * nproducers);
    }
#pragma omp barrier
  }
}

/**
 * TEST MPMC QUEUE
 * */

TEST_CASE("TestMpmcQueueInt") {
  Allocator *alloc = alloc_g;
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
  ProduceThenConsume<hipc::mpmc_queue<int>, int>(1, 1, 32);
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
}

TEST_CASE("TestMpmcQueueString") {
  Allocator *alloc = alloc_g;
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
  ProduceThenConsume<hipc::mpmc_queue<hipc::string>, hipc::string>(1, 1, 32);
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
}

TEST_CASE("TestMpmcQueueIntMultiThreaded") {
  Allocator *alloc = alloc_g;
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
  ProduceAndConsume<hipc::mpmc_queue<int>, int>(8, 1, 8192);
  // ProduceAndConsume<hipc::mpmc_queue<int>, int>(8, 8, 8192);
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
}

TEST_CASE("TestMpmcQueueStringMultiThreaded") {
  Allocator *alloc = alloc_g;
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
  ProduceAndConsume<hipc::mpmc_queue<hipc::string>, hipc::string>(8, 1, 8192);
  ProduceAndConsume<hipc::mpmc_queue<hipc::string>, hipc::string>(8, 8, 8192);
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
}
