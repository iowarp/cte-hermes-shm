#ifndef HSHM_SHM_INCLUDE_HSHM_SHM_DATA_STRUCTURES_IPC_RING_QUEUE_FLAGS_H_
#define HSHM_SHM_INCLUDE_HSHM_SHM_DATA_STRUCTURES_IPC_RING_QUEUE_FLAGS_H_

#include "hermes_shm/types/bitfield.h"
#include "hermes_shm/types/numbers.h"

namespace hshm::ipc {

typedef hshm::u32 RingQueueFlag;

/** Ring queue flags */
class RqFlag {
 public:
  CLS_CONST RingQueueFlag kNone = 0;
  /** Push should be atomic */
  CLS_CONST RingQueueFlag kPushAtomic = BIT_OPT(RingQueueFlag, 0);
  /** Pop should be atomic */
  CLS_CONST RingQueueFlag kPopAtomic = BIT_OPT(RingQueueFlag, 1);
  /** Queue is assumed to have fixed number of reqs */
  CLS_CONST RingQueueFlag kFixedReqs = BIT_OPT(RingQueueFlag, 2);
  /** Queue is circular */
  CLS_CONST RingQueueFlag kCircular = BIT_OPT(RingQueueFlag, 3);
};

/** Ring queue flag checks */
template <RingQueueFlag RQ_FLAGS>
class RqFlags {
 public:
  CLS_CONST RingQueueFlag IsPopAtomic = RQ_FLAGS & RqFlag::kPopAtomic;
  CLS_CONST RingQueueFlag IsPushAtomic = RQ_FLAGS & RqFlag::kPushAtomic;
  CLS_CONST RingQueueFlag HasFixedReqs = RQ_FLAGS & RqFlag::kFixedReqs;
  CLS_CONST RingQueueFlag IsCircular = RQ_FLAGS & RqFlag::kCircular;
};

}  // namespace hshm::ipc

namespace hshm {

using hshm::ipc::RingQueueFlag;
using hshm::ipc::RqFlag;
using hshm::ipc::RqFlags;

}  // namespace hshm

// bool IsPushAtomic,
// bool IsPopAtomic,
// bool HasFixedReqs,
#define RING_BUFFER_MPSC_FLAGS RqFlag::kPushAtomic | RqFlag::kPopAtomic
#define RING_BUFFER_SPSC_FLAGS 0
#define RING_BUFFER_FIXED_SPSC_FLAGS RqFlag::kFixedReqs
#define RING_BUFFER_FIXED_MPMC_FLAGS \
  RqFlag::kPushAtomic | RqFlag::kPopAtomic | RqFlag::kFixedReqs

#define RING_QUEUE_DEFS                                                  \
  CLS_CONST RingQueueFlag IsPopAtomic = RQ_FLAGS & RqFlag::kPopAtomic;   \
  CLS_CONST RingQueueFlag IsPushAtomic = RQ_FLAGS & RqFlag::kPushAtomic; \
  CLS_CONST RingQueueFlag HasFixedReqs = RQ_FLAGS & RqFlag::kFixedReqs;  \
  CLS_CONST RingQueueFlag IsCircular = RQ_FLAGS & RqFlag::kCircular;

#endif  // HSHM_SHM_INCLUDE_HSHM_SHM_DATA_STRUCTURES_IPC_RING_QUEUE_FLAGS_H_