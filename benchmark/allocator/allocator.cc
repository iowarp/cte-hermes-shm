/*
 * Copyright (C) 2022  SCS Lab <scslab@iit.edu>,
 * Luke Logan <llogan@hawk.iit.edu>,
 * Jaime Cernuda Garcia <jcernudagarcia@hawk.iit.edu>
 * Jay Lofstead <gflofst@sandia.gov>,
 * Anthony Kougkas <akougkas@iit.edu>,
 * Xian-He Sun <sun@iit.edu>
 *
 * This file is part of hermes
 *
 * hermes is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU General Public
 * License along with this program.  If not, see
 * <http://www.gnu.org/licenses/>.
 */

#include "basic_test.h"
#include "test_init.h"
#include "omp.h"

#include <string>
#include <hermes_shm/data_structures/string.h>

/** Test cases for the allocator */
class AllocatorTestSuite {
 public:
  std::string alloc_type_;
  Allocator *alloc_;
  static std::stringstream ss_;
  static int test_count_;

  ////////////////////
  /// Test Cases
  ////////////////////

  /** Constructor */
  AllocatorTestSuite(AllocatorType alloc_type, Allocator *alloc)
    : alloc_(alloc) {
    switch (alloc_type) {
      case AllocatorType::kStackAllocator: {
        alloc_type_ = "hipc::StackAllocator";
        break;
      }
      case AllocatorType::kPageAllocator: {
        alloc_type_ = "hipc::PageAllocator";
        break;
      }
      case AllocatorType::kMallocAllocator: {
        alloc_type_ = "hipc::MallocAllocator";
        break;
      }
      case AllocatorType::kMultiPageAllocator: {
        alloc_type_ = "hipc::MultiPageAllocator";
        break;
      }
    }
  }

  /** Allocate and Free a single size in a single loop */
  void AllocateAndFreeFixedSize(int count, size_t size) {
    Timer t;
    t.Resume();
    for (int i = 0; i < count; ++i) {
      Pointer p = alloc_->Allocate(size);
      alloc_->Free(p);
    }
#pragma omp barrier
    t.Pause();

    TestOutput("AllocateAndFreeFixedSize", size, t);
  }

  /** Allocate a fixed size in a loop, and then free in another loop */
  void AllocateThenFreeFixedSize(int count, size_t size) {
    Timer t;
    std::vector<Pointer> cache(count);
    t.Resume();
    for (int i = 0; i < count; ++i) {
      cache[i] = alloc_->Allocate(size);
    }
    for (int i = 0; i < count; ++i) {
      alloc_->Free(cache[i]);
    }
#pragma omp barrier
    t.Pause();

    TestOutput("AllocateThenFreeFixedSize", size, t);
  }

  /** Allocate, Free, Reallocate, Free in a loop */

  ////////////////////
  /// Test Output
  ////////////////////

  /** The CSV header */
  void TestOutputHeader() {
    ss_ << "test_name" << ","
        << "alloc_type" << ","
        << "alloc_size" << ","
        << "nthreads" << ","
        << "time" << std::endl;
  }

  /** The CSV test case */
  void TestOutput(const std::string &test_name, size_t obj_size, Timer &t) {
    int rank = omp_get_thread_num();
    if (rank != 0) { return; }
    if (test_count_ == 0) {
      TestOutputHeader();
    }
    ss_ << test_name << ","
        << alloc_type_ << ","
        << obj_size << ","
        << omp_get_num_threads() << ","
        << t.GetMsec() << std::endl;
    ++test_count_;
  }

  /** Print the CSV output */
  static void PrintTestOutput() {
    std::cout << ss_.str() << std::endl;
  }
};

/** The output text */
std::stringstream AllocatorTestSuite::ss_;

/** Number of tests currently conducted */
int AllocatorTestSuite::test_count_ = 0;

/** The minor number to use for allocators */
static int minor = 1;

/** Create the allocator + backend for the test */
template<typename BackendT, typename AllocT, typename ...Args>
Allocator* Pretest(MemoryBackendType backend_type,
                   Args&& ...args) {
  int rank = omp_get_thread_num();
  allocator_id_t alloc_id(0, minor);
  Allocator *alloc;
  auto mem_mngr = HERMES_MEMORY_MANAGER;

  if (rank == 0) {
    // Create the allocator + backend
    std::string shm_url = "test_allocators";
    mem_mngr->CreateBackend<BackendT>(
      MemoryManager::kDefaultBackendSize, shm_url);
    alloc = mem_mngr->CreateAllocator<AllocT>(
      shm_url, alloc_id, 0, std::forward<Args>(args)...);
  }
#pragma omp barrier
  if (rank != 0){
    // Retrieve the allocator + backend
    alloc = mem_mngr->GetAllocator(alloc_id);
  }

# pragma omp barrier
  if (rank == 0) {
    minor += 1;
  }

  return alloc;
}

/** Destroy the allocator + backend from the test */
void Posttest() {
  int rank = omp_get_thread_num();
#pragma omp barrier
  if (rank == 0) {
    std::string shm_url = "test_allocators";
    auto mem_mngr = HERMES_MEMORY_MANAGER;
    mem_mngr->DestroyBackend(shm_url);
  }
}

/** A series of allocator benchmarks for a particular thread */
template<typename BackendT, typename AllocT, typename ...Args>
void AllocatorTest(AllocatorType alloc_type,
                   MemoryBackendType backend_type,
                   Args&& ...args) {
  Allocator *alloc = Pretest<BackendT, AllocT>(
    backend_type, std::forward<Args>(args)...);
  // Allocate many, and then free many
  AllocatorTestSuite(alloc_type, alloc).AllocateThenFreeFixedSize(
    (2<<10), MEGABYTES(1));
  // Allocate and free immediately
  AllocatorTestSuite(alloc_type, alloc).AllocateAndFreeFixedSize(
    (2<<10), MEGABYTES(1));
  Posttest();
}

/** Test different allocators on a particular thread */
void FullAllocatorTestPerThread() {
  // Malloc allocator
  AllocatorTest<hipc::NullBackend, hipc::MallocAllocator>(
    AllocatorType::kMallocAllocator,
    MemoryBackendType::kNullBackend);/**/
  // Stack allocator
  AllocatorTest<hipc::PosixShmMmap, hipc::StackAllocator>(
    AllocatorType::kStackAllocator,
    MemoryBackendType::kPosixShmMmap);
  // Page allocator
  AllocatorTest<hipc::PosixShmMmap, hipc::PageAllocator>(
    AllocatorType::kPageAllocator,
    MemoryBackendType::kPosixShmMmap,
    MEGABYTES(1));
  // MultiPage allocator
  AllocatorTest<hipc::PosixShmMmap, hipc::MultiPageAllocator>(
    AllocatorType::kMultiPageAllocator,
    MemoryBackendType::kPosixShmMmap);
}

/** Spawn multiple threads and run allocator tests */
void FullAllocatorTestThreaded(int nthreads) {
  HERMES_THREAD_MANAGER->GetThreadStatic();

  omp_set_dynamic(0);
#pragma omp parallel num_threads(nthreads)
  {
#pragma omp barrier
    FullAllocatorTestPerThread();
#pragma omp barrier
  }
}

TEST_CASE("AllocatorBenchmark") {
  FullAllocatorTestThreaded(1);
  /*FullAllocatorTestThreaded(2);
  FullAllocatorTestThreaded(4);
  FullAllocatorTestThreaded(8);*/
  AllocatorTestSuite::PrintTestOutput();
}