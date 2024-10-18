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

#include "hermes_shm/thread/thread_model/thread_model_factory.h"
#include "hermes_shm/thread/thread_model/thread_model.h"
#include "hermes_shm/memory/memory_manager.h"

#ifdef HERMES_PTHREADS_ENABLED
#include "hermes_shm/thread/thread_model/pthread.h"
#endif
#ifdef HERMES_RPC_THALLIUM
#include "hermes_shm/thread/thread_model/argobots.h"
#endif
#include "hermes_shm/util/logging.h"

namespace hshm::thread_model {

HSHM_CROSS_FUN
ThreadModel* ThreadFactory::Get(ThreadType type) {
  switch (type) {
    case ThreadType::kPthread: {
#ifdef HERMES_PTHREADS_ENABLED
      return HERMES_MEMORY_MANAGER->GetDefaultAllocator()->NewObj<Pthread>();
#else
      return nullptr;
#endif
    }
    case ThreadType::kArgobots: {
#ifdef HERMES_RPC_THALLIUM
      return HERMES_MEMORY_MANAGER->GetDefaultAllocator()->NewObj<Argobots>();
#else
      return nullptr;
#endif
    }
    default: {
      HELOG(kWarning, "No such thread type");
      return nullptr;
    }
  }
}

}  // namespace hshm::thread_model
