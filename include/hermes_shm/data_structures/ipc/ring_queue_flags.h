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
  /** If out of space, wait for space. A different thread needs to pop. */
  CLS_CONST RingQueueFlag kWaitForSpace = BIT_OPT(RingQueueFlag, 3);
  /** If out of space, return null qtoken indicating error */
  CLS_CONST RingQueueFlag kErrorOnNoSpace = BIT_OPT(RingQueueFlag, 4);
};

}  // namespace hshm::ipc

namespace hshm {

using hshm::ipc::RingQueueFlag;
using hshm::ipc::RqFlag;

}  // namespace hshm

// bool IsPushAtomic,
// bool IsPopAtomic,
// bool HasFixedReqs,
#define RING_BUFFER_MPSC_FLAGS \
  RqFlag::kPushAtomic | RqFlag::kPopAtomic | RqFlag::kWaitForSpace
#define RING_BUFFER_SPSC_FLAGS RqFlag::kWaitForSpace
#define RING_BUFFER_FIXED_SPSC_FLAGS RqFlag::kErrorOnNoSpace
#define RING_BUFFER_FIXED_MPMC_FLAGS \
  RqFlag::kPushAtomic | RqFlag::kPopAtomic | RqFlag::kErrorOnNoSpace
#define RING_BUFFER_CIRCULAR_SPSC_FLAGS 0
#define RING_BUFFER_CIRCULAR_MPMC_FLAGS RqFlag::kPushAtomic | RqFlag::kPopAtomic

#define RING_QUEUE_DEFS                                                    \
  CLS_CONST RingQueueFlag IsPopAtomic = RQ_FLAGS & RqFlag::kPopAtomic;     \
  CLS_CONST RingQueueFlag IsPushAtomic = RQ_FLAGS & RqFlag::kPushAtomic;   \
  CLS_CONST RingQueueFlag WaitForSpace = RQ_FLAGS & RqFlag::kWaitForSpace; \
  CLS_CONST RingQueueFlag ErrorOnNoSpace = RQ_FLAGS & RqFlag::kErrorOnNoSpace;

#endif  // HSHM_SHM_INCLUDE_HSHM_SHM_DATA_STRUCTURES_IPC_RING_QUEUE_FLAGS_H_