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

#include <random>
#include <string>

#include "basic_test.h"
#include "hermes_shm/data_structures/ipc/string.h"
#include "omp.h"
#include "test_init.h"

/** Test cases for the allocator */
template <typename AllocT>
class AllocatorTestSuite {
 public:
  std::string alloc_type_;
  hipc::CtxAllocator<AllocT> alloc_;
  Timer timer_;

  /**====================================
   * Test Runner
   * ===================================*/

  /** Constructor */
  AllocatorTestSuite(AllocatorType alloc_type,
                     hipc::CtxAllocator<AllocT> &alloc)
      : alloc_(alloc) {
    switch (alloc_type) {
      case AllocatorType::kStackAllocator: {
        alloc_type_ = "hipc::StackAllocator";
        break;
      }
      case AllocatorType::kMallocAllocator: {
        alloc_type_ = "hipc::MallocAllocator";
        break;
      }
      case AllocatorType::kFixedPageAllocator: {
        alloc_type_ = "hipc::FixedPageAllocator";
        break;
      }
      case AllocatorType::kScalablePageAllocator: {
        alloc_type_ = "hipc::ScalablePageAllocator";
        break;
      }
      case AllocatorType::kThreadLocalAllocator: {
        alloc_type_ = "hipc::ThreadLocalAllocator";
        break;
      }
      case AllocatorType::kTestAllocator: {
        alloc_type_ = "hipc::TestAllocator";
        break;
      }
      default: {
        HELOG(kFatal, "Could not find this allocator type");
        break;
      }
    }
  }

  /**====================================
   * Test Cases
   * ===================================*/

  /** Allocate and Free a single size in a single loop */
  void AllocateAndFreeFixedSize(size_t count, size_t size) {
    StartTimer();
    for (size_t i = 0; i < count; ++i) {
      Pointer p = alloc_->Allocate(alloc_.ctx_, size);
      alloc_->Free(alloc_.ctx_, p);
    }
    StopTimer();

    TestOutput("AllocateAndFreeFixedSize", size, count, timer_);
  }

  /** Allocate a fixed size in a loop, and then free in another loop */
  void AllocateThenFreeFixedSize(size_t count, size_t size) {
    StartTimer();
    std::vector<Pointer> cache(count);
    for (size_t i = 0; i < count; ++i) {
      cache[i] = alloc_->Allocate(alloc_.ctx_, size);
    }
    for (size_t i = 0; i < count; ++i) {
      alloc_->Free(alloc_.ctx_, cache[i]);
    }
    StopTimer();

    TestOutput("AllocateThenFreeFixedSize", count, size, timer_);
  }

  void seq(std::vector<size_t> &vec, size_t rep, size_t count) {
    for (size_t i = 0; i < count; ++i) {
      vec.emplace_back(rep);
    }
  }

  /** Allocate a window of pages, free the window. Random page sizes. */
  void AllocateAndFreeRandomWindow(size_t count) {
    std::mt19937 rng(23522523);
    std::vector<size_t> sizes_;

    seq(sizes_, 64, hshm::Unit<size_t>::Megabytes(1) / 64);
    seq(sizes_, 190, hshm::Unit<size_t>::Megabytes(1) / 190);
    seq(sizes_, hshm::Unit<size_t>::Kilobytes(1),
        hshm::Unit<size_t>::Megabytes(1) / hshm::Unit<size_t>::Kilobytes(1));
    seq(sizes_, hshm::Unit<size_t>::Kilobytes(4),
        hshm::Unit<size_t>::Megabytes(8) / hshm::Unit<size_t>::Kilobytes(4));
    seq(sizes_, hshm::Unit<size_t>::Kilobytes(32),
        hshm::Unit<size_t>::Megabytes(4) / hshm::Unit<size_t>::Kilobytes(4));
    seq(sizes_, hshm::Unit<size_t>::Megabytes(1),
        hshm::Unit<size_t>::Megabytes(64) / hshm::Unit<size_t>::Megabytes(1));
    std::shuffle(std::begin(sizes_), std::end(sizes_), rng);
    std::vector<Pointer> window(sizes_.size());
    size_t num_windows = 500;

    StartTimer();
    for (size_t w = 0; w < num_windows; ++w) {
      for (size_t i = 0; i < sizes_.size(); ++i) {
        auto &size = sizes_[i];
        window[i] = alloc_->Allocate(alloc_.ctx_, size);
      }
      for (size_t i = 0; i < sizes_.size(); ++i) {
        alloc_->Free(alloc_.ctx_, window[i]);
      }
    }
    StopTimer();

    TestOutput("AllocateAndFreeRandomWindow", 0, count, timer_);
  }

  /**====================================
   * Test Helpers
   * ===================================*/

  void StartTimer() {
    int rank = omp_get_thread_num();
    if (rank == 0) {
      timer_.Reset();
      timer_.Resume();
    }
#pragma omp barrier
  }

  void StopTimer() {
#pragma omp barrier
    int rank = omp_get_thread_num();
    if (rank == 0) {
      timer_.Pause();
    }
  }

  /** The CSV test case */
  void TestOutput(const std::string &test_name, size_t obj_size,
                  size_t count_per_rank, Timer &t) {
    int rank = omp_get_thread_num();
    if (rank != 0) {
      return;
    }
    int nthreads = omp_get_num_threads();
    double count = (double)count_per_rank * nthreads;
    HILOG(kInfo, "{},{},{},{},{},{} ms,{} KOps", test_name, alloc_type_,
          obj_size, nthreads, count, t.GetMsec(), count / t.GetMsec());
  }

  /** Print the CSV output */
  static void PrintTestHeader() {
    HILOG(kInfo, "test_name,alloc_type,obj_size,msec,nthreads,count,KOps");
  }
};

/** The minor number to use for allocators */
static int minor = 0;
const std::string shm_url = "test_allocators";

/** Create the allocator + backend for the test */
template <typename BackendT, typename AllocT, typename... Args>
AllocT *Pretest(MemoryBackendType backend_type, Args &&...args) {
  int rank = omp_get_thread_num();
  AllocatorId alloc_id(1, minor);
  auto mem_mngr = HSHM_MEMORY_MANAGER;

  if (rank == 0) {
    // Create the allocator + backend
    mem_mngr->UnregisterAllocator(alloc_id);
    mem_mngr->UnregisterBackend(hipc::MemoryBackendId::Get(0));
    mem_mngr->CreateBackendWithUrl<BackendT>(hipc::MemoryBackendId::Get(0),
                                             mem_mngr->GetDefaultBackendSize(),
                                             shm_url);
    mem_mngr->CreateAllocator<AllocT>(hipc::MemoryBackendId::Get(0), alloc_id,
                                      0, std::forward<Args>(args)...);
  }
#pragma omp barrier

  auto *alloc = mem_mngr->GetAllocator<AllocT>(alloc_id);
  if (alloc == nullptr) {
    HELOG(kFatal, "Failed to find the memory allocator?");
  }
  return alloc;
}

/** Destroy the allocator + backend from the test */
void Posttest() {
  int rank = omp_get_thread_num();
#pragma omp barrier
  if (rank == 0) {
    AllocatorId alloc_id(1, minor);
    HSHM_MEMORY_MANAGER->UnregisterAllocator(alloc_id);
    HSHM_MEMORY_MANAGER->UnregisterBackend(hipc::MemoryBackendId::Get(0));
    minor += 1;
  }
#pragma omp barrier
}

/** A series of allocator benchmarks for a particular thread */
template <typename BackendT, typename AllocT, typename... Args>
void AllocatorTest(AllocatorType alloc_type, MemoryBackendType backend_type,
                   Args &&...args) {
  auto *alloc =
      Pretest<BackendT, AllocT>(backend_type, std::forward<Args>(args)...);
  PAGE_DIVIDE("Test") {
    hipc::ScopedTlsAllocator<AllocT> scoped_tls(alloc);
    //  if (alloc_type == AllocatorType::kScalablePageAllocator) {
    //    printf("TID: %llu\n", (*scoped_tls).ctx_.tid_);
    //  }
    size_t count = (1 << 16);
    // Allocate many and then free many
    //  AllocatorTestSuite<AllocT>((alloc_type,
    //  *scoped_tls).AllocateThenFreeFixedSize(
    //    count, hshm::Unit<size_t>::Kilobytes(1));
    // Allocate and free immediately
    AllocatorTestSuite<AllocT>(alloc_type, *scoped_tls)
        .AllocateAndFreeFixedSize(count, hshm::Unit<size_t>::Kilobytes(1));
    // Allocate and free randomly
    // AllocatorTestSuite<AllocT>(alloc_type, *scoped_tls)
    //     .AllocateAndFreeRandomWindow(count);
  }
  Posttest();
}

/** Test different allocators on a particular thread */
void FullAllocatorTestPerThread() {
  // Malloc allocator
  AllocatorTest<hipc::MallocBackend, hipc::MallocAllocator>(
      AllocatorType::kMallocAllocator, MemoryBackendType::kMallocBackend);
  // Scalable page allocator
  AllocatorTest<hipc::PosixShmMmap, hipc::ScalablePageAllocator>(
      AllocatorType::kScalablePageAllocator, MemoryBackendType::kMallocBackend);
  // Thread-local allocator
  AllocatorTest<hipc::PosixShmMmap, hipc::ThreadLocalAllocator>(
      AllocatorType::kThreadLocalAllocator, MemoryBackendType::kMallocBackend);
  // Test allocator
  AllocatorTest<hipc::PosixShmMmap, hipc::TestAllocator>(
      AllocatorType::kTestAllocator, MemoryBackendType::kMallocBackend);
  // Stack allocator
  //  AllocatorTest<hipc::PosixShmMmap, hipc::StackAllocator>(
  //    AllocatorType::kStackAllocator,
  //    MemoryBackendType::kPosixShmMmap);
}

/** Spawn multiple threads and run allocator tests */
void FullAllocatorTestThreaded(int nthreads) {
  omp_set_dynamic(0);
#pragma omp parallel num_threads(nthreads)
  {
#pragma omp barrier
    try {
      FullAllocatorTestPerThread();
    } catch (const std::exception &e) {
      HELOG(kFatal, "Exception: {}", e.what());
    } catch (hshm::Error &e) {
      HELOG(kFatal, "Error: {}", e.what());
    }
#pragma omp barrier
  }
}

TEST_CASE("AllocatorBenchmark") {
  HSHM_ERROR_HANDLE_START();
  AllocatorTestSuite<hipc::NullAllocator>::PrintTestHeader();
  FullAllocatorTestThreaded(1);
  // FullAllocatorTestThreaded(2);
  // FullAllocatorTestThreaded(4);
  // FullAllocatorTestThreaded(8);
  // FullAllocatorTestThreaded(16);
  HSHM_ERROR_HANDLE_END();
}
