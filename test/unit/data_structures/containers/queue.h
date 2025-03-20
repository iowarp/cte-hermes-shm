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

#ifndef HSHM_SHM_TEST_UNIT_DATA_STRUCTURES_CONTAINERS_QUEUE_H_
#define HSHM_SHM_TEST_UNIT_DATA_STRUCTURES_CONTAINERS_QUEUE_H_

#include "basic_test.h"
#include "hermes_shm/data_structures/all.h"
#include "hermes_shm/types/numbers.h"
#include "hermes_shm/util/logging.h"
#include "test_init.h"

#ifdef HSHM_ENABLE_OPENMP
#include <omp.h>
#endif

struct IntEntry : public hipc::list_queue_entry {
  int value;

  /** Default constructor */
  IntEntry() : value(0) {}

  /** Constructor */
  explicit IntEntry(int val) : value(val) {}
};

template <typename NewT>
class VariableMaker {
 public:
  std::vector<NewT> vars_;
  hipc::atomic<hshm::size_t> count_;

 public:
  explicit VariableMaker(size_t total_vars) : vars_(total_vars) { count_ = 0; }

  static NewT _MakeVariable(size_t num) {
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
      STATIC_ASSERT(false, "Unsupported type", NewT);
    }
  }

  NewT MakeVariable(size_t num) {
    NewT var = _MakeVariable(num);
    size_t count = count_.fetch_add(1);
    vars_[count] = var;
    return var;
  }

  size_t GetIntFromVar(NewT &var) {
    if constexpr (std::is_arithmetic_v<NewT>) {
      return static_cast<size_t>(var);
    } else if constexpr (std::is_same_v<NewT, std::string>) {
      return std::stoi(var);
    } else if constexpr (std::is_same_v<NewT, hipc::string>) {
      return std::stoi(var.str());
    } else if constexpr (std::is_same_v<NewT, IntEntry *>) {
      return var->value;
    } else {
      STATIC_ASSERT(false, "Unsupported type", NewT);
    }
  }

  void FreeVariable(NewT &var) {
    if constexpr (std::is_same_v<NewT, IntEntry *>) {
      auto alloc = HSHM_DEFAULT_ALLOC;
      alloc->DelObj(HSHM_DEFAULT_MEM_CTX, var);
    }
  }

  void FreeVariables() {
    if constexpr (std::is_same_v<NewT, IntEntry *>) {
      size_t count = count_.load();
      for (size_t i = 0; i < count; ++i) {
        auto alloc = HSHM_DEFAULT_ALLOC;
        alloc->DelObj(HSHM_DEFAULT_MEM_CTX, vars_[i]);
      }
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
  void Produce(VariableMaker<T> &var_maker, size_t count_per_rank) {
    std::vector<size_t> idxs;
    int rank = omp_get_thread_num();
    try {
      for (size_t i = 0; i < count_per_rank; ++i) {
        size_t idx = rank * count_per_rank + i;
        T var = var_maker.MakeVariable(idx);
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
  void Consume(int min_rank, VariableMaker<T> &var_maker,
               std::atomic<size_t> &count, size_t total_count,
               std::vector<size_t> &entries) {
    T entry;
    // Consume everything
    while (count < total_count) {
      auto qtok = queue_.pop(entry);
      if (qtok.IsNull()) {
        continue;
      }
      size_t entry_int = var_maker.GetIntFromVar(entry);
      size_t off = count.fetch_add(1);
      if (off >= total_count) {
        break;
      }
      entries[off] = entry_int;
      // var_maker.FreeVariable(entry);
    }

    int rank = omp_get_thread_num();
    HILOG(kInfo, "Rank {}: Consumed {} entries", rank, count.load());
    if (rank == min_rank) {
      // Ensure there's no data left in the queue
      REQUIRE(queue_.pop(entry).IsNull());
      // Ensure the data is all correct
      REQUIRE(entries.size() == total_count);
      std::sort(entries.begin(), entries.end());
      REQUIRE(entries.size() == total_count);
      for (size_t i = 0; i < total_count; ++i) {
        REQUIRE(entries[i] == i);
      }
      var_maker.FreeVariables();
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
  VariableMaker<T> var_maker(nproducers * count_per_rank);
  entries.resize(count_per_rank * nproducers);

  // Produce all the data
  omp_set_dynamic(0);
#pragma omp parallel shared(var_maker, nproducers, count_per_rank, q, count, \
                                entries) num_threads(nproducers)  // NOLINT
  {                                                               // NOLINT
#pragma omp barrier
    q.Produce(var_maker, count_per_rank);
#pragma omp barrier
  }

  omp_set_dynamic(0);
#pragma omp parallel shared(var_maker, nproducers, count_per_rank, q) \
    num_threads(nconsumers)  // NOLINT
  {                          // NOLINT
#pragma omp barrier
     // Consume all the data
    q.Consume(0, var_maker, count, count_per_rank * nproducers, entries);
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
  VariableMaker<T> var_maker(nproducers * count_per_rank);
  entries.resize(count_per_rank * nproducers);

  // Produce all the data
  omp_set_dynamic(0);
#pragma omp parallel shared(var_maker, nproducers, count_per_rank, q, count) \
    num_threads(nthreads)  // NOLINT
  {                        // NOLINT
#pragma omp barrier
    size_t rank = omp_get_thread_num();
    if (rank < nproducers) {
      // Producer
      q.Produce(var_maker, count_per_rank);
    } else {
      // Consumer
      q.Consume(nproducers, var_maker, count, count_per_rank * nproducers,
                entries);
    }
#pragma omp barrier
  }
}

#endif  // HSHM_SHM_TEST_UNIT_DATA_STRUCTURES_CONTAINERS_QUEUE_H_
