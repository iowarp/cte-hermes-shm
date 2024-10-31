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

void PageAllocationTest(Allocator *alloc) {
  size_t count = 1024;
  size_t page_size = KILOBYTES(4);
  auto mem_mngr = HERMES_MEMORY_MANAGER;

  // Allocate pages
  std::vector<Pointer> ps(count);
  void *ptrs[count];
  for (size_t i = 0; i < count; ++i) {
    ptrs[i] = alloc->AllocatePtr<void>(
        hshm::ThreadId::GetNull(), page_size, ps[i]);
    memset(ptrs[i], i, page_size);
    REQUIRE(ps[i].off_.load() != 0);
    REQUIRE(!ps[i].IsNull());
    REQUIRE(ptrs[i] != nullptr);
  }

  // Convert process pointers into independent pointers
  for (size_t i = 0; i < count; ++i) {
    Pointer p = mem_mngr->Convert(ptrs[i]);
    REQUIRE(p == ps[i]);
    REQUIRE(VerifyBuffer((char*)ptrs[i], page_size, i));
  }

  // Check the custom header
  auto hdr = alloc->GetCustomHeader<SimpleAllocatorHeader>();
  REQUIRE(hdr->checksum_ == HEADER_CHECKSUM);

  // Free pages
  for (size_t i = 0; i < count; ++i) {
    alloc->Free(hshm::ThreadId::GetNull(), ps[i]);
  }

  // Reallocate pages
  for (size_t i = 0; i < count; ++i) {
    ptrs[i] = alloc->AllocatePtr<void>(
        hshm::ThreadId::GetNull(), page_size, ps[i]);
    REQUIRE(ps[i].off_.load() != 0);
    REQUIRE(!ps[i].IsNull());
  }

  // Free again
  for (size_t i = 0; i < count; ++i) {
    alloc->Free(hshm::ThreadId::GetNull(), ps[i]);
  }

  return;
}

void MultiPageAllocationTest(Allocator *alloc) {
  std::vector<size_t> alloc_sizes = {
    64, 128, 256,
    KILOBYTES(1), KILOBYTES(4), KILOBYTES(64),
    MEGABYTES(1)
  };

  // Allocate and free pages between 64 bytes and 32MB
  {
    for (size_t r = 0; r < 4; ++r) {
      for (size_t i = 0; i < alloc_sizes.size(); ++i) {
        Pointer ps[16];
        for (size_t j = 0; j < 16; ++j) {
          ps[j] = alloc->Allocate(
              hshm::ThreadId::GetNull(), alloc_sizes[i]);
        }
        for (size_t j = 0; j < 16; ++j) {
          alloc->Free(hshm::ThreadId::GetNull(), ps[j]);
        }
      }
    }
  }
}

void ReallocationTest(Allocator *alloc) {
  std::vector<std::pair<size_t, size_t>> sizes = {
      {KILOBYTES(3), KILOBYTES(4)},
      {KILOBYTES(4), MEGABYTES(1)}
  };

  // Reallocate a small page to a larger page
  for (auto &[small_size, large_size] : sizes) {
    Pointer p;
    char *ptr = alloc->AllocatePtr<char>(
        hshm::ThreadId::GetNull(), small_size, p);
    memset(ptr, 10, small_size);
    char *new_ptr = alloc->ReallocatePtr<char>(
        hshm::ThreadId::GetNull(), p, large_size);
    for (size_t i = 0; i < small_size; ++i) {
      REQUIRE(ptr[i] == 10);
    }
    memset(new_ptr, 0, large_size);
    alloc->Free(hshm::ThreadId::GetNull(), p);
  }
}

void AlignedAllocationTest(Allocator *alloc) {
  std::vector<std::pair<size_t, size_t>> sizes = {
      {KILOBYTES(4), KILOBYTES(4)},
  };

  // Aligned allocate pages
  for (auto &[size, alignment] : sizes) {
    for (size_t i = 0; i < 1024; ++i) {
      Pointer p;
      char *ptr = alloc->AllocatePtr<char>(
          hshm::ThreadId::GetNull(), size, p, alignment);
      REQUIRE(((size_t)ptr % alignment) == 0);
      memset(alloc->Convert<void>(p), 0, size);
      alloc->Free(hshm::ThreadId::GetNull(), p);
    }
  }
}

TEST_CASE("StackAllocator") {
  auto alloc = Pretest<hipc::PosixShmMmap, hipc::StackAllocator>();
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
  PageAllocationTest(alloc);
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
  Posttest();
}

TEST_CASE("MallocAllocator") {
  auto alloc = Pretest<hipc::MallocBackend, hipc::MallocAllocator>();
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
  PageAllocationTest(alloc);
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);

  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
  MultiPageAllocationTest(alloc);
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);

  Posttest();
}

TEST_CASE("ScalablePageAllocator") {
  auto alloc = Pretest<hipc::PosixShmMmap, hipc::ScalablePageAllocator>();
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
  PageAllocationTest(alloc);
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);

  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
  MultiPageAllocationTest(alloc);
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);

  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
  ReallocationTest(alloc);
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);

  Posttest();
}

TEST_CASE("LocalPointers") {
  auto alloc = Pretest<hipc::PosixShmMmap, hipc::ScalablePageAllocator>();
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
  // Allocate API
  hipc::LPointer<char> p1 =
      alloc->AllocateLocalPtr<char>(hshm::ThreadId::GetNull(), 256);
  REQUIRE(!p1.shm_.IsNull());
  REQUIRE(p1.ptr_ != nullptr);
  hipc::LPointer<char> p2 =
      alloc->ClearAllocateLocalPtr<char>(hshm::ThreadId::GetNull(), 256);
  REQUIRE(!p2.shm_.IsNull());
  REQUIRE(p2.ptr_ != nullptr);
  REQUIRE(*p2 == 0);
  hipc::LPointer<char> p3 =
      alloc->ReallocateLocalPtr<char>(hshm::ThreadId::GetNull(), p1, 256);
  REQUIRE(!p3.shm_.IsNull());
  REQUIRE(p3.ptr_ != nullptr);
  alloc->FreeLocalPtr(hshm::ThreadId::GetNull(), p1);
  alloc->FreeLocalPtr(hshm::ThreadId::GetNull(), p3);

  // OBJ API
  hipc::LPointer<std::vector<int>> p4 =
      alloc->NewObjLocal<std::vector<int>>(hshm::ThreadId::GetNull());
  alloc->DelObjLocal(hshm::ThreadId::GetNull(), p4);
  hipc::LPointer<std::vector<int>> p5 =
      alloc->NewObjsLocal<std::vector<int>>(
          hshm::ThreadId::GetNull(), 4);
  alloc->ReallocateObjsLocal<std::vector<int>>(
      hshm::ThreadId::GetNull(), p5, 5);
  alloc->ReallocateConstructObjsLocal<std::vector<int>>(
      hshm::ThreadId::GetNull(), p5, 4, 5);
  alloc->DelObjsLocal(hshm::ThreadId::GetNull(), p5, 5);
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
  Posttest();
}

TEST_CASE("Arrays") {
  auto alloc = Pretest<hipc::PosixShmMmap, hipc::ScalablePageAllocator>();
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
  // Allocate API
  hipc::Array p1 = alloc->AllocateArray<char>(
      hshm::ThreadId::GetNull(), 256);
  REQUIRE(!p1.shm_.IsNull());
  hipc::Array p2 = alloc->ClearAllocateArray<char>(
      hshm::ThreadId::GetNull(), 256);
  REQUIRE(!p2.shm_.IsNull());
  hipc::Array p3 = alloc->ReallocateArray<char>(
      hshm::ThreadId::GetNull(), p1, 256);
  REQUIRE(!p3.shm_.IsNull());
  alloc->FreeArray(hshm::ThreadId::GetNull(), p1);
  alloc->FreeArray(hshm::ThreadId::GetNull(), p3);
  Posttest();
}

TEST_CASE("LocalArrays") {
  auto alloc = Pretest<hipc::PosixShmMmap, hipc::ScalablePageAllocator>();
  REQUIRE(alloc->GetCurrentlyAllocatedSize() == 0);
  // Allocate API
  hipc::LArray<char> p1 = alloc->AllocateLocalArray<char>(
      hshm::ThreadId::GetNull(), 256);
  REQUIRE(!p1.shm_.IsNull());
  REQUIRE(p1.ptr_ != nullptr);
  hipc::LArray<char> p2 = alloc->ClearAllocateLocalArray<char>(
      hshm::ThreadId::GetNull(), 256);
  REQUIRE(!p2.shm_.IsNull());
  REQUIRE(p2.ptr_ != nullptr);
  REQUIRE(*p2 == 0);
  hipc::LArray<char> p3 = alloc->ReallocateLocalArray<char>(
      hshm::ThreadId::GetNull(), p1, 256);
  REQUIRE(!p3.shm_.IsNull());
  REQUIRE(p3.ptr_ != nullptr);
  alloc->FreeLocalArray(hshm::ThreadId::GetNull(), p1);
  alloc->FreeLocalArray(hshm::ThreadId::GetNull(), p3);
}
