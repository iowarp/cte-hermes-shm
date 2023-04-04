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

#ifndef HERMES_THREAD_PTHREAD_H_
#define HERMES_THREAD_PTHREAD_H_

#include "thread.h"
#include <errno.h>
#include <hermes_shm/util/errors.h>
#include <omp.h>
#include "hermes_shm/introspect/system_info.h"

namespace hshm {

template<typename BIND> class Pthread;

/** Parameters passed to the pthread */
template<typename FUNC>
struct PthreadParams {
  FUNC func_;
  Pthread<FUNC> *pthread_;

  /** Default constructor */
  PthreadParams(Pthread<FUNC> *pthread, FUNC bind) :
    func_(bind),
    pthread_(pthread) {}
};

template<typename FUNC>
class Pthread : public Thread {
 private:
  pthread_t pthread_;

 public:
  bool started_;

 public:
  /** Default constructor */
  Pthread() = default;

  /** Spawn constructor */
  explicit Pthread(FUNC bind) : pthread_(-1), started_(false) {
    PthreadParams<FUNC> params(this, bind);
    int ret = pthread_create(&pthread_, nullptr,
                             DoWork, &params);
    if (ret != 0) {
      throw PTHREAD_CREATE_FAILED.format();
    }
    while (!started_) {}
  }

  /** Pause a thread */
  void Pause() override {}

  /** Resume a thread */
  void Resume() override {}

  /** Join the thread */
  void Join() override {
    void *ret;
    pthread_join(pthread_, &ret);
  }

  /** Set thread affinity to the mask */
  void SetAffinity(const cpu_bitfield &mask) override {
    int ncpu = HERMES_SYSTEM_INFO->ncpu_;
    cpu_set_t cpus[ncpu];
    for (size_t i = 0; i < mask.size(); ++i) {
      memcpy((void*)&cpus[i],
             (void*)&mask.bits_[i],
             sizeof(bitfield32_t));
    }
    pthread_setaffinity_np_safe(ncpu, cpus);
  }

  /** Get thread affinity according to the mask */
  void GetAffinity(cpu_bitfield &mask) override {
    pthread_getaffinity_np(pthread_,
                           CPU_SETSIZE,
                           (cpu_set_t*)mask.bits_);
  }

  /** Yield the thread for a period of time */
  void SleepForUs(size_t us) override {
    usleep(us);
  }

  /** Yield thread time slice */
  void Yield() override {
    sched_yield();
  }

  /** Get the TID of the current thread */
  tid_t GetTid() override {
    return omp_get_thread_num();
    // return static_cast<tid_t>(pthread_self());
  }

 private:
  /** Execute the function */
  static void* DoWork(void *void_params) {
    auto params = (*reinterpret_cast<PthreadParams<FUNC>*>(void_params));
    params.pthread_->started_ = true;
    if constexpr(!std::is_same_v<FUNC, int>) {
      params.func_();
    }
    return nullptr;
  }

  /** Get the CPU affinity of this thread */
  inline void pthread_setaffinity_np_safe(int n_cpu, cpu_set_t *cpus) {
    int ret = pthread_setaffinity_np(pthread_, n_cpu, cpus);
    if (ret != 0) {
      throw INVALID_AFFINITY.format(strerror(ret));
    }
  }
};

}  // namespace hshm

#endif  // HERMES_THREAD_PTHREAD_H_
