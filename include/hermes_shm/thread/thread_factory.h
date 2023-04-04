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

#ifndef HERMES_THREAD_THREAD_FACTORY_H_
#define HERMES_THREAD_THREAD_FACTORY_H_

#include "thread.h"
#include "pthread.h"
#include "argobots.h"

namespace hshm {

template<typename FUNC=int>
class ThreadFactory {
 private:
  ThreadType type_;
  FUNC func_;
#ifdef HERMES_RPC_THALLIUM
  ABT_xstream *xstream_;
#endif


 public:
  /** Create a thread without spawning */
  explicit ThreadFactory(ThreadType type) : type_(type) {}

  /** Create and spawn a thread */
  explicit ThreadFactory(ThreadType type, FUNC func)
  : type_(type), func_(func) {}

#ifdef HERMES_RPC_THALLIUM
  /** Create and spawn a thread (argobots) */
  explicit ThreadFactory(ThreadType type, ABT_xstream &xstream, FUNC func)
    : xstream_(&xstream), type_(type), func_(func) {}
#endif

  /**  */
  std::unique_ptr<Thread> Get() {
    switch (type_) {
      case ThreadType::kPthread: {
#ifdef HERMES_PTHREADS_ENABLED
        return std::make_unique<Pthread<FUNC>>(func_);
#else
        return nullptr;
#endif
      }
      case ThreadType::kArgobots: {
#ifdef HERMES_RPC_THALLIUM
        return std::make_unique<Argobots<FUNC>>(*xstream_, func_);
#else
        return nullptr;
#endif
      }
      default: return nullptr;
    }
  }
};

}  // namespace hshm

#endif  // HERMES_THREAD_THREAD_FACTORY_H_
