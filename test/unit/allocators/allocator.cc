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
  hipc::FullPtr<int> ret = alloc->NewObjLocal<int>(HSHM_DEFAULT_MEM_CTX);
  hipc::FullPtr<int> ret2(ret.ptr_);
  REQUIRE(ret == ret2);
  hipc::FullPtr<int> ret3(ret.shm_);
  REQUIRE(ret == ret3);
  alloc->DelObjLocal(HSHM_DEFAULT_MEM_CTX, ret);
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
  hipc::FullPtr<char> p1 =
      alloc->AllocateLocalPtr<char>(HSHM_DEFAULT_MEM_CTX, 256);
  REQUIRE(!p1.shm_.IsNull());
  REQUIRE(p1.ptr_ != nullptr);
  hipc::FullPtr<char> p2 =
      alloc->ClearAllocateLocalPtr<char>(HSHM_DEFAULT_MEM_CTX, 256);
  REQUIRE(!p2.shm_.IsNull());
  REQUIRE(p2.ptr_ != nullptr);
  REQUIRE(*p2 == 0);
  alloc->ReallocateLocalPtr<char>(HSHM_DEFAULT_MEM_CTX, p1, 256);
  REQUIRE(!p1.shm_.IsNull());
  REQUIRE(p1.ptr_ != nullptr);
  alloc->FreeLocalPtr(HSHM_DEFAULT_MEM_CTX, p1);
  alloc->FreeLocalPtr(HSHM_DEFAULT_MEM_CTX, p2);

  // OBJ API
  hipc::FullPtr<std::vector<int>> p4 =
      alloc->NewObjLocal<std::vector<int>>(HSHM_DEFAULT_MEM_CTX);
  alloc->DelObjLocal(HSHM_DEFAULT_MEM_CTX, p4);
  hipc::FullPtr<std::vector<int>> p5 =
      alloc->NewObjsLocal<std::vector<int>>(HSHM_DEFAULT_MEM_CTX, 4);
  alloc->ReallocateObjsLocal<std::vector<int>>(HSHM_DEFAULT_MEM_CTX, p5, 5);
  alloc->ReallocateConstructObjsLocal<std::vector<int>>(HSHM_DEFAULT_MEM_CTX,
                                                        p5, 4, 5);
  alloc->DelObjsLocal(HSHM_DEFAULT_MEM_CTX, p5, 5);
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
  Posttest();
}

TEST_CASE("Arrays") {
  auto alloc = Pretest<hipc::PosixShmMmap, hipc::ScalablePageAllocator>();
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
  // Allocate API
  hipc::Array p1 = alloc->AllocateArray<char>(HSHM_DEFAULT_MEM_CTX, 256);
  REQUIRE(!p1.shm_.IsNull());
  hipc::Array p2 = alloc->ClearAllocateArray<char>(HSHM_DEFAULT_MEM_CTX, 256);
  REQUIRE(!p2.shm_.IsNull());
  alloc->ReallocateArray<char>(HSHM_DEFAULT_MEM_CTX, p1, 256);
  REQUIRE(!p1.shm_.IsNull());
  alloc->FreeArray(HSHM_DEFAULT_MEM_CTX, p1);
  Posttest();
}

TEST_CASE("LocalArrays") {
  auto alloc = Pretest<hipc::PosixShmMmap, hipc::ScalablePageAllocator>();
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
  // Allocate API
  hipc::LArray<char> p1 =
      alloc->AllocateLocalArray<char>(HSHM_DEFAULT_MEM_CTX, 256);
  REQUIRE(!p1.shm_.IsNull());
  REQUIRE(p1.ptr_ != nullptr);
  hipc::LArray<char> p2 =
      alloc->ClearAllocateLocalArray<char>(HSHM_DEFAULT_MEM_CTX, 256);
  REQUIRE(!p2.shm_.IsNull());
  REQUIRE(p2.ptr_ != nullptr);
  REQUIRE(*p2 == 0);
  alloc->ReallocateLocalArray<char>(HSHM_DEFAULT_MEM_CTX, p1, 256);
  REQUIRE(!p1.shm_.IsNull());
  REQUIRE(p1.ptr_ != nullptr);
  alloc->FreeLocalArray(HSHM_DEFAULT_MEM_CTX, p1);
}
