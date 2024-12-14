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

#ifndef HERMES_SHM_TEST_UNIT_DATA_STRUCTURES_CONTAINERS_QUEUE_H_
#define HERMES_SHM_TEST_UNIT_DATA_STRUCTURES_CONTAINERS_QUEUE_H_

#include "hermes_shm/data_structures/all.h"
#include "omp.h"
#include "test_init.h"

struct IntEntry : public hipc::iqueue_entry {
  int value;

  /** Default constructor */
  IntEntry() : value(0) {}

  /** Constructor */
  explicit IntEntry(int val) : value(val) {}
};

template <typename NewT>
class VariableMaker {
 public:
  static NewT MakeVariable(size_t num) {
    if constexpr (std::is_arithmetic_v<NewT>) {
      return static_cast<NewT>(num);
    } else if constexpr (std::is_same_v<NewT, std::string>) {
      return std::to_string(num);
    } else if constexpr (std::is_same_v<NewT, hipc::string>) {
      return hipc::string(std::to_string(num));
    } else if constexpr (std::is_same_v<NewT, IntEntry *>) {
      auto alloc = HSHM_DEFAULT_ALLOC;
      return alloc->template NewObjLocal<IntEntry>(HSHM_DEFAULT_MEM_CTX, num)
          .ptr_;
    } else {
      static_assert(false, "Unsupported type");
    }
  }

  static size_t GetIntFromVar(NewT &var) {
    if constexpr (std::is_arithmetic_v<NewT>) {
      return static_cast<size_t>(var);
    } else if constexpr (std::is_same_v<NewT, std::string>) {
      return std::stoi(var);
    } else if constexpr (std::is_same_v<NewT, hipc::string>) {
      return std::stoi(var.str());
    } else if constexpr (std::is_same_v<NewT, IntEntry *>) {
      return var->value;
    } else {
      static_assert(false, "Unsupported type");
    }
  }

  static void FreeVariable(NewT &var) {
    if constexpr (std::is_same_v<NewT, IntEntry *>) {
      auto alloc = HSHM_DEFAULT_ALLOC;
      alloc->DelObj(HSHM_DEFAULT_MEM_CTX, var);
    }
  }
};

template <typename QueueT, typename T>
class QueueTestSuite {
 public:
  QueueT &queue_;

 public:
  /** Constructor */
  explicit QueueTestSuite(QueueT &queue) : queue_(queue) {}

  /** Producer method */
  void Produce(size_t count_per_rank) {
    std::vector<size_t> idxs;
    int rank = omp_get_thread_num();
    try {
      for (size_t i = 0; i < count_per_rank; ++i) {
        size_t idx = rank * count_per_rank + i;
        T var = VariableMaker<T>::MakeVariable(idx);
        idxs.emplace_back(idx);
        while (queue_.emplace(var).IsNull()) {
        }
      }
    } catch (hshm::Error &e) {
      HELOG(kFatal, e.what());
    }
    REQUIRE(idxs.size() == count_per_rank);
    std::sort(idxs.begin(), idxs.end());
    for (size_t i = 0; i < count_per_rank; ++i) {
      size_t idx = rank * count_per_rank + i;
      REQUIRE(idxs[i] == idx);
    }
  }

  /** Consumer method */
  void Consume(std::atomic<size_t> &count, size_t total_count,
               std::vector<size_t> &entries) {
    T entry;
    // Consume everything
    while (count < total_count) {
      auto qtok = queue_.pop(entry);
      if (qtok.IsNull()) {
        continue;
      }
      size_t entry_int = VariableMaker<T>::GetIntFromVar(entry);
      size_t off = count.fetch_add(1);
      if (off >= total_count) {
        break;
      }
      entries[off] = entry_int;
      VariableMaker<T>::FreeVariable(entry);
    }

    int rank = omp_get_thread_num();
    if (rank == 0) {
      // Ensure there's no data left in the queue
      REQUIRE(queue_.pop(entry).IsNull());
      // Ensure the data is all correct
      REQUIRE(entries.size() == total_count);
      std::sort(entries.begin(), entries.end());
      REQUIRE(entries.size() == total_count);
      for (size_t i = 0; i < total_count; ++i) {
        REQUIRE(entries[i] == i);
      }
    }
  }
};

template <typename QueueT, typename T>
void ProduceThenConsume(size_t nproducers, size_t nconsumers,
                        size_t count_per_rank, size_t depth) {
  QueueT queue(depth);
  QueueTestSuite<QueueT, T> q(queue);
  std::atomic<size_t> count = 0;
  std::vector<size_t> entries;
  entries.resize(count_per_rank * nproducers);

  // Produce all the data
  omp_set_dynamic(0);
#pragma omp parallel shared(nproducers, count_per_rank, q, count, entries) \
    num_threads(nproducers)  // NOLINT
  {                          // NOLINT
#pragma omp barrier
    q.Produce(count_per_rank);
#pragma omp barrier
  }

  omp_set_dynamic(0);
#pragma omp parallel shared(nproducers, count_per_rank, q) \
    num_threads(nconsumers)  // NOLINT
  {                          // NOLINT
#pragma omp barrier
     // Consume all the data
    q.Consume(count, count_per_rank * nproducers, entries);
#pragma omp barrier
  }
}

template <typename QueueT, typename T>
void ProduceAndConsume(size_t nproducers, size_t nconsumers,
                       size_t count_per_rank, size_t depth) {
  QueueT queue(depth);
  size_t nthreads = nproducers + nconsumers;
  QueueTestSuite<QueueT, T> q(queue);
  std::atomic<size_t> count = 0;
  std::vector<size_t> entries;
  entries.resize(count_per_rank * nproducers);

  // Produce all the data
  omp_set_dynamic(0);
#pragma omp parallel shared(nproducers, count_per_rank, q, count) \
    num_threads(nthreads)  // NOLINT
  {                        // NOLINT
#pragma omp barrier
    size_t rank = omp_get_thread_num();
    if (rank < nproducers) {
      // Producer
      q.Produce(count_per_rank);
    } else {
      // Consumer
      q.Consume(count, count_per_rank * nproducers, entries);
    }
#pragma omp barrier
  }
}

#endif  // HERMES_SHM_TEST_UNIT_DATA_STRUCTURES_CONTAINERS_QUEUE_H_
