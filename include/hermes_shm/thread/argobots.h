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

#ifndef HERMES_SHM_INCLUDE_HERMES_SHM_THREAD_THALLIUM_H_
#define HERMES_SHM_INCLUDE_HERMES_SHM_THREAD_THALLIUM_H_

#include "thread.h"
#include <thallium.hpp>
#include <errno.h>
#include <hermes_shm/util/errors.h>
#include <omp.h>
#include "hermes_shm/introspect/system_info.h"

namespace hshm {

template<typename FUNC> class Argobots;

/** Parameters passed to the pthread */
template<typename FUNC>
struct ArgobotsParams {
  FUNC func_;
  Argobots<FUNC> *obj_;

  /** Default constructor */
  ArgobotsParams(Argobots<FUNC> *obj, FUNC bind) :
    func_(bind),
    obj_(obj) {}
};

template<typename FUNC>
class Argobots : public Thread {
 private:
  ABT_thread thread_;
  ABT_key key_;
  ArgobotsParams<FUNC> params_;

 public:
  bool started_;

 public:
  /** Default constructor */
  Argobots() = default;

  /** Spawn constructor */
  explicit Argobots(ABT_xstream &xstream, FUNC func)
  : params_(this, func), started_(false) {
    ABT_thread_create_on_xstream(xstream,
                                 DoWork, (void*)&params_,
                                 ABT_THREAD_ATTR_NULL, &thread_);
    while(!started_);
  }

  /** Pause a thread */
  void Pause() override {}

  /** Resume a thread */
  void Resume() override {}

  /** Join the thread */
  void Join() override {
    ABT_thread_join(thread_);
  }

  /** Set thread affinity to the mask */
  void SetAffinity(const cpu_bitfield &mask) override {
    // TODO(llogan)
  }

  /** Get thread affinity according to the mask */
  void GetAffinity(cpu_bitfield &mask) override {
    // TODO(llogan)
  }

  /** Yield the thread for a period of time */
  void SleepForUs(size_t us) override {
    usleep(us);
  }

  /** Yield thread time slice */
  void Yield() override {
    ABT_thread_yield();
  }

  /** Get the TID of the current thread */
  tid_t GetTid() override {
    return ABT_thread_self(&thread_);
  }

 private:
  /** Execute the function */
  static void DoWork(void *void_params) {
    auto params = (*reinterpret_cast<ArgobotsParams<FUNC>*>(void_params));
    // Start working
    params.obj_->started_ = true;
    if constexpr(!std::is_same_v<FUNC, int>) {
      params.func_();
    }
  }
};

}  // namespace hshm

#endif //HERMES_SHM_INCLUDE_HERMES_SHM_THREAD_THALLIUM_H_
