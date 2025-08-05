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

#ifndef HSHM_TEST_UNIT_ALLOCATORS_TEST_INIT_H_
#define HSHM_TEST_UNIT_ALLOCATORS_TEST_INIT_H_

#include "basic_test.h"
// hermes_shm/memory/memory_manager.h is now included via hermes_shm.h in basic_test.h

using hshm::ipc::Allocator;
using hshm::ipc::AllocatorId;
using hshm::ipc::AllocatorType;
using hshm::ipc::MemoryBackend;
using hshm::ipc::MemoryBackendType;
using hshm::ipc::MemoryManager;
using hshm::ipc::Pointer;

#define HEADER_CHECKSUM 8482942

struct SimpleAllocatorHeader {
  int checksum_;
};

template <typename BackendT, typename AllocT>
AllocT *Pretest() {
  std::string shm_url = "test_allocators";
  AllocatorId alloc_id(1, 0);
  auto mem_mngr = HSHM_MEMORY_MANAGER;
  mem_mngr->UnregisterAllocator(alloc_id);
  mem_mngr->DestroyBackend(hipc::MemoryBackendId::Get(0));
  mem_mngr->CreateBackendWithUrl<BackendT>(
      hipc::MemoryBackendId::Get(0), hshm::Unit<size_t>::Gigabytes(1), shm_url);
  mem_mngr->CreateAllocator<AllocT>(hipc::MemoryBackendId::Get(0), alloc_id,
                                    sizeof(SimpleAllocatorHeader));
  auto alloc = mem_mngr->GetAllocator<AllocT>(alloc_id);
  auto hdr = alloc->template GetCustomHeader<SimpleAllocatorHeader>();
  hdr->checksum_ = HEADER_CHECKSUM;
  return alloc;
}

void Posttest();

template <typename AllocT>
class Workloads {
 public:
  static void PageAllocationTest(AllocT *alloc) {
    size_t count = 1024;
    size_t page_size = hshm::Unit<size_t>::Kilobytes(4);
    auto mem_mngr = HSHM_MEMORY_MANAGER;

    // Allocate pages
    std::vector<Pointer> ps(count);
    std::vector<void *> ptrs(count);
    for (size_t i = 0; i < count; ++i) {
      ptrs[i] = alloc->template AllocatePtr<void>(HSHM_DEFAULT_MEM_CTX,
                                                  page_size, ps[i]);
      memset(ptrs[i], i, page_size);
      REQUIRE(ps[i].off_.load() != 0);
      REQUIRE(!ps[i].IsNull());
      REQUIRE(ptrs[i] != nullptr);
    }

    // Convert process pointers into independent pointers
    for (size_t i = 0; i < count; ++i) {
      Pointer p = mem_mngr->Convert(ptrs[i]);
      REQUIRE(p.alloc_id_.bits_.major_ == ps[i].alloc_id_.bits_.major_);
      REQUIRE(VerifyBuffer((char *)ptrs[i], page_size, (char)i));
    }

    // Check the custom header
    auto hdr = alloc->template GetCustomHeader<SimpleAllocatorHeader>();
    REQUIRE(hdr->checksum_ == HEADER_CHECKSUM);

    // Free pages
    for (size_t i = 0; i < count; ++i) {
      alloc->Free(HSHM_DEFAULT_MEM_CTX, ps[i]);
    }

    // Reallocate pages
    for (size_t i = 0; i < count; ++i) {
      ptrs[i] = alloc->template AllocatePtr<void>(HSHM_DEFAULT_MEM_CTX,
                                                  page_size, ps[i]);
      REQUIRE(ps[i].off_.load() != 0);
      REQUIRE(!ps[i].IsNull());
    }

    // Free again
    for (size_t i = 0; i < count; ++i) {
      alloc->Free(HSHM_DEFAULT_MEM_CTX, ps[i]);
    }

    return;
  }

  static void MultiPageAllocationTest(AllocT *alloc) {
    std::vector<size_t> alloc_sizes = {64,
                                       128,
                                       256,
                                       hshm::Unit<size_t>::Kilobytes(1),
                                       hshm::Unit<size_t>::Kilobytes(4),
                                       hshm::Unit<size_t>::Kilobytes(64),
                                       hshm::Unit<size_t>::Megabytes(1)};

    // Allocate and free pages between 64 bytes and 32MB
    {
      for (size_t r = 0; r < 4; ++r) {
        for (size_t i = 0; i < alloc_sizes.size(); ++i) {
          Pointer ps[16];
          for (size_t j = 0; j < 16; ++j) {
            ps[j] = alloc->Allocate(HSHM_DEFAULT_MEM_CTX, alloc_sizes[i]);
          }
          for (size_t j = 0; j < 16; ++j) {
            alloc->Free(HSHM_DEFAULT_MEM_CTX, ps[j]);
          }
        }
      }
    }
  }

  static void ReallocationTest(AllocT *alloc) {
    std::vector<std::pair<size_t, size_t>> sizes = {
        {hshm::Unit<size_t>::Kilobytes(3), hshm::Unit<size_t>::Kilobytes(4)},
        {hshm::Unit<size_t>::Kilobytes(4), hshm::Unit<size_t>::Megabytes(1)}};

    // Reallocate a small page to a larger page
    for (auto &[small_size, large_size] : sizes) {
      Pointer p;
      char *ptr = alloc->template AllocatePtr<char>(HSHM_DEFAULT_MEM_CTX,
                                                    (size_t)small_size, p);
      memset(ptr, 10, small_size);
      char *new_ptr = alloc->template ReallocatePtr<char>(HSHM_DEFAULT_MEM_CTX,
                                                          p, large_size);
      for (size_t i = 0; i < small_size; ++i) {
        REQUIRE(ptr[i] == 10);
      }
      memset(new_ptr, 0, large_size);
      alloc->Free(HSHM_DEFAULT_MEM_CTX, p);
    }
  }

  static void AlignedAllocationTest(AllocT *alloc) {
    std::vector<std::pair<size_t, size_t>> sizes = {
        {hshm::Unit<size_t>::Kilobytes(4), hshm::Unit<size_t>::Kilobytes(4)},
    };

    // Aligned allocate pages
    for (auto &[size, alignment] : sizes) {
      for (size_t i = 0; i < 1024; ++i) {
        Pointer p;
        char *ptr = alloc->template AllocatePtr<char>(HSHM_DEFAULT_MEM_CTX,
                                                      size, p, alignment);
        REQUIRE(((size_t)ptr % alignment) == 0);
        memset(alloc->template Convert<void>(p), 0, size);
        alloc->Free(HSHM_DEFAULT_MEM_CTX, p);
      }
    }
  }
};

#endif  // HSHM_TEST_UNIT_ALLOCATORS_TEST_INIT_H_
