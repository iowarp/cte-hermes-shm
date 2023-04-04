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
#include "hermes_shm/data_structures/ipc/mpsc_queue.h"

template<typename T>
class QueueTest {
 public:
  hipc::uptr<hipc::mpsc_queue<int>> &queue_;

 public:
  /** Constructor */
  explicit QueueTest(hipc::uptr<hipc::mpsc_queue<int>> &queue)
  : queue_(queue) {}

  /** Producer method */
  void Produce(size_t count_per_rank) {
    int rank = omp_get_thread_num();
    for (size_t i = 0; i < count_per_rank; ++i) {
      size_t idx = rank * count_per_rank + i;
      CREATE_SET_VAR_TO_INT_OR_STRING(T, var, idx);
      queue_->emplace(var);
    }
  }

  /** Consumer method */
  void Consume(size_t nthreads, size_t count_per_rank) {
    std::vector<int> entries;
    size_t total_size = nthreads * count_per_rank;
    entries.reserve(count_per_rank);
    auto entry = hipc::make_uptr<T>();
    auto entry_ref = hipc::to_ref(entry);

    while(entries.size() < total_size) {
      bool success = queue_->pop(entry_ref);
      if (!success) {
        HERMES_THREAD_MODEL->Yield();
      }
      CREATE_GET_INT_FROM_VAR(T, entry_int, entry_ref)
      REQUIRE(entries.size() < total_size);
      entries.emplace_back(entry_int);
    }
  }
};

TEST_CASE("MpscTest") {
  size_t depth = 1024;
  auto queue = hipc::make_uptr<hipc::mpsc_queue<int>>(depth);

  // Emplace 1024 elements

  // Pop 1024 elements
}

TEST_CASE("MpscTestMultiThreaded") {
}