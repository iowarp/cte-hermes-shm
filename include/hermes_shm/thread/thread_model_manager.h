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

#ifndef HSHM_THREAD_THREAD_MANAGER_H_
#define HSHM_THREAD_THREAD_MANAGER_H_

#include "hermes_shm/constants/macros.h"
#include "hermes_shm/introspect/system_info.h"
#include "hermes_shm/thread/thread_model/thread_model.h"

#ifdef HSHM_ENABLE_PTHREADS
#include "thread_model/pthread.h"
#endif
#ifdef HSHM_RPC_THALLIUM
#include "thread_model/argobots.h"
#endif
#ifdef HSHM_ENABLE_CUDA
#include "thread_model/cuda.h"
#endif
#ifdef HSHM_ENABLE_ROCM
#include "thread_model/rocm.h"
#endif
#ifdef HSHM_ENABLE_WINDOWS_THREADS
#include "thread_model/windows.h"
#endif

#include "hermes_shm/util/singleton.h"

#if defined(HSHM_IS_HOST)
#define HSHM_THREAD_MODEL \
  hshm::CrossSingleton<HSHM_DEFAULT_THREAD_MODEL>::GetInstance()
#define HSHM_THREAD_MODEL_T hshm::HSHM_DEFAULT_THREAD_MODEL*
#elif defined(HSHM_IS_GPU)
#define HSHM_THREAD_MODEL \
  hshm::CrossSingleton<HSHM_DEFAULT_THREAD_MODEL_GPU>::GetInstance()
#define HSHM_THREAD_MODEL_T hshm::HSHM_DEFAULT_THREAD_MODEL_GPU*
#endif

#endif  // HSHM_THREAD_THREAD_MANAGER_H_
