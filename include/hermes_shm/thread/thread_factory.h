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

namespace hshm {

template<typename BIND=int>
class ThreadFactory {
 private:
  ThreadType type_;
  BIND bind_;

 public:
  /** Create a thread without spawning */
  explicit ThreadFactory(ThreadType type) : type_(type) {}

  /** Create and spawn a thread */
  explicit ThreadFactory(ThreadType type, BIND bind)
  : type_(type), bind_(bind) {}

  /**  */
  std::unique_ptr<Thread> Get() {
    switch (type_) {
      case ThreadType::kPthread: {
        return std::make_unique<Pthread<BIND>>(bind_);
      }
      default: return nullptr;
    }
  }
};

}  // namespace hshm

#endif  // HERMES_THREAD_THREAD_FACTORY_H_
