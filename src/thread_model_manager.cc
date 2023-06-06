//
// Created by lukemartinlogan on 6/6/23.
//

#include "hermes_shm/thread/thread_model_manager.h"
#include "hermes_shm/util/logging.h"

namespace hshm {

/** Set the threading model of this application */
void ThreadModelManager::SetThreadModel(ThreadType type) {
  static std::mutex lock_;
  lock_.lock();
  if (type_ == type) {
    lock_.unlock();
    return;
  }
  type_ = type;
  thread_static_ = thread_model::ThreadFactory::Get(type);
  if (thread_static_ == nullptr) {
    HELOG(kFatal, "Could not load the threading model");
  }
  lock_.unlock();
}

/** Sleep for a period of time (microseconds) */
void ThreadModelManager::SleepForUs(size_t us) {
  thread_static_->SleepForUs(us);
}

/** Call Yield */
void ThreadModelManager::Yield() {
  thread_static_->Yield();
}

/** Call GetTid */
tid_t ThreadModelManager::GetTid() {
  return thread_static_->GetTid();
}

}  // namespace hshm