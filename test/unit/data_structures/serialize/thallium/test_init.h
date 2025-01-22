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

#ifndef HSHM_SHM_TEST_UNIT_DATA_STRUCTURES_SERIALIZE_THALLIUM_TEST_INIT_H_
#define HSHM_SHM_TEST_UNIT_DATA_STRUCTURES_SERIALIZE_THALLIUM_TEST_INIT_H_

#include <thallium.hpp>

#include "hermes_shm/data_structures/all.h"
#include "hermes_shm/thread/thread_model_manager.h"
#include "thallium.h"

using hshm::ipc::Allocator;
using hshm::ipc::AllocatorId;
using hshm::ipc::AllocatorType;
using hshm::ipc::MemoryBackend;
using hshm::ipc::MemoryBackendType;
using hshm::ipc::Pointer;
using hshm::ipc::PosixShmMmap;

using hshm::ipc::Allocator;
using hshm::ipc::AllocatorId;
using hshm::ipc::AllocatorType;
using hshm::ipc::MemoryBackend;
using hshm::ipc::MemoryBackendType;
using hshm::ipc::MemoryManager;
using hshm::ipc::Pointer;

namespace thallium {
class Constants {
 public:
  CLS_CONST char *kServerName = "ofi+sockets://127.0.0.1:8080";
  CLS_CONST char *kTestString = "012344823723642364723874623";

  /** Test cases */
  CLS_CONST char *kStringTest0 = "kStringTest0";
  CLS_CONST char *kStringTestLarge = "kStringTestLarge";
  CLS_CONST char *kCharbufTest0 = "kCharbufTest0";
  CLS_CONST char *kCharbufTestLarge = "kCharbufTestLarge";
  CLS_CONST char *kVecOfInt0Test = "kVecOfInt0Test";
  CLS_CONST char *kVecOfIntLargeTest = "kVecOfIntLargeTest";
  CLS_CONST char *kVecOfString0Test = "kVecOfString0Test";
  CLS_CONST char *kVecOfStringLargeTest = "kVecOfStringLargeTest";
  CLS_CONST char *kBitfieldTest = "kBitfieldTest";
  CLS_CONST char *kShmArTest = "kShmArTest";
};
}  // namespace thallium
using tcnst = thallium::Constants;

namespace tl = thallium;
using thallium::request;

/** Test init */
template <typename AllocT>
void ServerPretest() {
  std::string shm_url = "test_serializers";
  AllocatorId alloc_id(1, 0);
  auto mem_mngr = HSHM_MEMORY_MANAGER;
  mem_mngr->UnregisterAllocator(alloc_id);
  mem_mngr->DestroyBackend(hipc::MemoryBackendId::GetRoot());
  mem_mngr->CreateBackend<PosixShmMmap>(hipc::MemoryBackendId::Get(0),
                                        hshm::Unit<size_t>::Megabytes(100),
                                        shm_url);
  mem_mngr->CreateAllocator<AllocT>(hipc::MemoryBackendId::Get(0), alloc_id, 0);
}

template <typename AllocT>
void ClientPretest() {
  std::string shm_url = "test_serializers";
  AllocatorId alloc_id(1, 0);
  auto mem_mngr = HSHM_MEMORY_MANAGER;
  mem_mngr->AttachBackend(MemoryBackendType::kPosixShmMmap, shm_url);
}

extern std::unique_ptr<tl::engine> client_;
extern std::unique_ptr<tl::engine> server_;

#endif  // HSHM_SHM_TEST_UNIT_DATA_STRUCTURES_SERIALIZE_THALLIUM_TEST_INIT_H_
