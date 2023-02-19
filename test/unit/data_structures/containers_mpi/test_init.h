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


#ifndef HERMES_TEST_UNIT_DATA_STRUCTURES_TEST_INIT_H_
#define HERMES_TEST_UNIT_DATA_STRUCTURES_TEST_INIT_H_

#include "hermes_shm/memory/allocator/page_allocator.h"
#include "hermes_shm/data_structures/data_structure.h"
#include <mpi.h>

using hermes::ipc::PosixShmMmap;
using hermes::ipc::MemoryBackendType;
using hermes::ipc::MemoryBackend;
using hermes::ipc::allocator_id_t;
using hermes::ipc::AllocatorType;
using hermes::ipc::Allocator;
using hermes::ipc::Pointer;

using hermes::ipc::MemoryBackendType;
using hermes::ipc::MemoryBackend;
using hermes::ipc::allocator_id_t;
using hermes::ipc::AllocatorType;
using hermes::ipc::Allocator;
using hermes::ipc::MemoryManager;
using hermes::ipc::Pointer;

extern Allocator *alloc_g;

void Posttest();

#endif  // HERMES_TEST_UNIT_DATA_STRUCTURES_TEST_INIT_H_
