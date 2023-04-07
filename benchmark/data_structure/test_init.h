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

#ifndef HERMES_BENCHMARK_DATA_STRUCTURE_TEST_INIT_H_
#define HERMES_BENCHMARK_DATA_STRUCTURE_TEST_INIT_H_

#include <boost/interprocess/allocators/allocator.hpp>
#include <boost/interprocess/containers/string.hpp>
#include <boost/interprocess/containers/vector.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/allocators/allocator.hpp>

#include <boost/container/scoped_allocator.hpp>

#include "hermes_shm/data_structures/data_structure.h"
#include <hermes_shm/util/timer.h>
#include <hermes_shm/util/type_switch.h>

using hshm::ipc::MemoryBackendType;
using hshm::ipc::MemoryBackend;
using hshm::ipc::allocator_id_t;
using hshm::ipc::AllocatorType;
using hshm::ipc::Allocator;
using hshm::ipc::Pointer;

using hshm::ipc::MemoryBackendType;
using hshm::ipc::MemoryBackend;
using hshm::ipc::allocator_id_t;
using hshm::ipc::AllocatorType;
using hshm::ipc::Allocator;
using hshm::ipc::MemoryManager;
using hshm::ipc::Pointer;

namespace bipc = boost::interprocess;

namespace boost::interprocess {
/** Shared memory segment (a large contiguous region) */
typedef bipc::managed_shared_memory::segment_manager segment_manager_t;

/** A generic allocator */
typedef boost::container::scoped_allocator_adaptor<
bipc::allocator<void, segment_manager_t>>
  void_allocator;

/** A generic string using that allocator */
typedef boost::interprocess::basic_string<
  char, std::char_traits<char>, void_allocator> ipc_string;
}  // namespace bopost::interprocess

/** Instance of the allocator */
extern std::unique_ptr<bipc::void_allocator> alloc_inst_g;

/** Instance of the segment */
extern std::unique_ptr<bipc::managed_shared_memory> segment_g;

/** Timer */
using Timer = hshm::HighResMonotonicTimer;

#endif //HERMES_BENCHMARK_DATA_STRUCTURE_TEST_INIT_H_
