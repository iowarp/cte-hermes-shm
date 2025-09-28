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

TEST_CASE("FullPtr") {
  hipc::FullPtr<int> x;
  hipc::FullPtr<std::string> y;
  y = x.Cast<std::string>();

  auto alloc = Pretest<hipc::PosixShmMmap, hipc::StackAllocator>();
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
  hipc::FullPtr<int> ret = alloc->NewObj<int>(HSHM_DEFAULT_MEM_CTX);
  hipc::FullPtr<int> ret2(ret.ptr_);
  REQUIRE(ret == ret2);
  hipc::FullPtr<int> ret3(ret.shm_);
  REQUIRE(ret == ret3);
  alloc->DelObj(HSHM_DEFAULT_MEM_CTX, ret);
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
}

TEST_CASE("StackAllocator") {
  auto alloc = Pretest<hipc::PosixShmMmap, hipc::StackAllocator>();
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
  Workloads<hipc::StackAllocator>::PageAllocationTest(alloc);
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
  Posttest();
}

TEST_CASE("MallocAllocator") {
  auto alloc = Pretest<hipc::MallocBackend, hipc::MallocAllocator>();
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
  Workloads<hipc::MallocAllocator>::PageAllocationTest(alloc);
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);

  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
  Workloads<hipc::MallocAllocator>::MultiPageAllocationTest(alloc);
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);

  Posttest();
}

TEST_CASE("ScalablePageAllocator") {
  auto alloc = Pretest<hipc::PosixShmMmap, hipc::ScalablePageAllocator>();
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
  Workloads<hipc::ScalablePageAllocator>::PageAllocationTest(alloc);
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);

  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
  Workloads<hipc::ScalablePageAllocator>::MultiPageAllocationTest(alloc);
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);

  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
  Workloads<hipc::ScalablePageAllocator>::ReallocationTest(alloc);
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);

  Posttest();
}

TEST_CASE("ThreadLocalAllocator") {
  auto alloc = Pretest<hipc::PosixShmMmap, hipc::ThreadLocalAllocator>();
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
  Workloads<hipc::ThreadLocalAllocator>::PageAllocationTest(alloc);
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);

  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
  Workloads<hipc::ThreadLocalAllocator>::MultiPageAllocationTest(alloc);
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);

  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
  Workloads<hipc::ThreadLocalAllocator>::ReallocationTest(alloc);
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);

  Posttest();
}

TEST_CASE("LocaFullPtrs") {
  auto alloc = Pretest<hipc::PosixShmMmap, hipc::ScalablePageAllocator>();
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
  // Allocate API
  auto full_ptr1 = alloc->Allocate(HSHM_DEFAULT_MEM_CTX, 256);
  hipc::FullPtr<char> p1(reinterpret_cast<char*>(full_ptr1.ptr_), full_ptr1.shm_);
  REQUIRE(!p1.shm_.IsNull());
  REQUIRE(p1.ptr_ != nullptr);
  auto full_ptr2 = alloc->Allocate(HSHM_DEFAULT_MEM_CTX, 256);
  hipc::FullPtr<char> p2(reinterpret_cast<char*>(full_ptr2.ptr_), full_ptr2.shm_);
  memset(p2.ptr_, 0, 256); // ClearAllocate equivalent
  REQUIRE(!p2.shm_.IsNull());
  REQUIRE(p2.ptr_ != nullptr);
  REQUIRE(*p2 == 0);
  hipc::FullPtr<void> old_p1(reinterpret_cast<void*>(p1.ptr_), p1.shm_);
  auto reallocated = alloc->Reallocate(HSHM_DEFAULT_MEM_CTX, old_p1, 256);
  p1.ptr_ = reinterpret_cast<char*>(reallocated.ptr_);
  p1.shm_ = reallocated.shm_;
  REQUIRE(!p1.shm_.IsNull());
  REQUIRE(p1.ptr_ != nullptr);
  hipc::FullPtr<void> void_p1(reinterpret_cast<void*>(p1.ptr_), p1.shm_);
  alloc->Free(HSHM_DEFAULT_MEM_CTX, void_p1);
  hipc::FullPtr<void> void_p2(reinterpret_cast<void*>(p2.ptr_), p2.shm_);
  alloc->Free(HSHM_DEFAULT_MEM_CTX, void_p2);

  // OBJ API
  hipc::FullPtr<std::vector<int>> p4 =
      alloc->NewObj<std::vector<int>>(HSHM_DEFAULT_MEM_CTX);
  alloc->DelObj(HSHM_DEFAULT_MEM_CTX, p4);
  hipc::FullPtr<std::vector<int>> p5 =
      alloc->NewObjs<std::vector<int>>(HSHM_DEFAULT_MEM_CTX, 4);
  alloc->ReallocateObjs<std::vector<int>>(HSHM_DEFAULT_MEM_CTX, p5, 5);
  alloc->ConstructObjs<std::vector<int>>(p5.ptr_, 4, 5);
  alloc->DelObjs(HSHM_DEFAULT_MEM_CTX, p5, 5);
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
  Posttest();
}


