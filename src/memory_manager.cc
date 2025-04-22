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

#define HSHM_COMPILING_DLL
#define __HSHM_IS_COMPILING__

#include "hermes_shm/memory/memory_manager.h"

#include "hermes_shm/introspect/system_info.h"
#include "hermes_shm/memory/allocator/allocator_factory.h"
#include "hermes_shm/memory/backend/memory_backend_factory.h"
#include "hermes_shm/thread/thread_model_manager.h"
#include "hermes_shm/util/errors.h"
#include "hermes_shm/util/logging.h"

namespace hshm::ipc {

HSHM_DEFINE_GLOBAL_CROSS_PTR_VAR_CC(hshm::ipc::MemoryManager,
                                    hshmMemoryManager);

}  // namespace hshm::ipc

// TODO(llogan): Fix. A hack for HIP compiler to function
// I would love to spend more time figuring out why ROCm
// Fails to compile without this, but whatever.
#include "hermes_shm/introspect/system_info.cc"
