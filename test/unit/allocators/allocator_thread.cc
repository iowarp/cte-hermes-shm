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


#include "test_init.h"

template <typename AllocT>
void MultiThreadedPageAllocationTest(AllocT *alloc) {
  size_t nthreads = 8;
  omp_set_dynamic(0);
#pragma omp parallel shared(alloc) num_threads(nthreads)
  {
#pragma omp barrier
    Workloads<AllocT>::PageAllocationTest(alloc);
#pragma omp barrier
    try {
      Workloads<AllocT>::MultiPageAllocationTest(alloc);
    } catch (std::shared_ptr<hshm::Error> &err) {
      err->print();
      exit(1);
    }
#pragma omp barrier
  }
}

TEST_CASE("StackAllocatorMultithreaded") {
  HERMES_ERROR_HANDLE_START()
  auto alloc = Pretest<hipc::PosixShmMmap, hipc::StackAllocator>();
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
  MultiThreadedPageAllocationTest(alloc);
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
  Posttest();
  HERMES_ERROR_HANDLE_END()
}

TEST_CASE("ScalablePageAllocatorMultithreaded") {
  HERMES_ERROR_HANDLE_START()
  auto alloc = Pretest<hipc::PosixShmMmap, hipc::ScalablePageAllocator>();
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
  MultiThreadedPageAllocationTest(alloc);
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
  Posttest();
  HERMES_ERROR_HANDLE_END()
}
