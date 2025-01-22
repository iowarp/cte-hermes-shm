//
// Created by lukemartinlogan on 6/6/23.
//

#include "hermes_shm/hermes_shm.h"

int main() {
  std::string shm_url = "test_serializers";
  hipc::AllocatorId alloc_id(1, 0);
  auto mem_mngr = HSHM_MEMORY_MANAGER;
  mem_mngr->UnregisterAllocator(alloc_id);
  mem_mngr->UnregisterBackend(hipc::MemoryBackendId::Get(0));
  mem_mngr->CreateBackend<hipc::PosixShmMmap>(
      hipc::MemoryBackendId::Get(0), hshm::Unit<size_t>::Megabytes(100),
      shm_url);
  mem_mngr->CreateAllocator<hipc::StackAllocator>(hipc::MemoryBackendId::Get(0),
                                                  alloc_id, 0);
}