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

/**
 * This file should be included ONLY in the thread_model.cc file.
 * This file should be considered private to that class's implementation.
 * */

#ifndef HERMES_THREAD_THREAD_FACTORY_H_
#define HERMES_THREAD_THREAD_FACTORY_H_

#include "thread_model.h"
#ifdef HERMES_PTHREADS_ENABLED
#include "hermes_shm/thread/thread_model/pthread.h"
#endif
#ifdef HERMES_RPC_THALLIUM
#include "hermes_shm/thread/thread_model/argobots.h"
#endif
#ifdef HERMES_ENABLE_CUDA
#include "hermes_shm/thread/thread_model/cuda.h"
#endif
#include "hermes_shm/util/logging.h"

namespace hshm::thread_model {

#define HERMES_THREAD_FACTORY_CASE(T)\
  case ThreadType::k##T: {\
    new ((T*)buffer) T();\
    return (T*)buffer;\
  }

//#define HERMES_THREAD_FACTORY_CASE(T)\
//  case ThreadType::k##T: {           \
//    return HERMES_MEMORY_MANAGER->GetRootAllocator()->NewObj<T>(); \
//  }


class ThreadFactory {
 public:
  /** Get a thread instance */
  HSHM_CROSS_FUN
  static ThreadModel* Get(char *buffer, ThreadType type) {
    switch (type) {
#ifndef __CUDA_ARCH__
      ///////////// OFF GPU
#ifdef HERMES_PTHREADS_ENABLED
      HERMES_THREAD_FACTORY_CASE(Pthread)
#endif
#ifdef HERMES_RPC_THALLIUM
      HERMES_THREAD_FACTORY_CASE(Argobots)
#endif
#else
      ///////////// ON GPU
    HERMES_THREAD_FACTORY_CASE(Cuda)
#endif
      default: {
        HELOG(kWarning, "No such thread type");
        return nullptr;
      }
    }
  }
};

}  // namespace hshm::thread_model

#endif  // HERMES_THREAD_THREAD_FACTORY_H_
