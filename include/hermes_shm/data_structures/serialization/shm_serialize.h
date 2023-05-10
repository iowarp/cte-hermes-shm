//
// Created by llogan on 5/9/23.
//

#ifndef HERMES_SHM_INCLUDE_HERMES_SHM_DATA_STRUCTURES_SERIALIZATION_SHM_SERIALIZE_H_
#define HERMES_SHM_INCLUDE_HERMES_SHM_DATA_STRUCTURES_SERIALIZATION_SHM_SERIALIZE_H_

namespace hshm::ipc {

class ShmSerializer {
 public:
  size_t off_;

  /** Default constructor */
  ShmSerializer() : off_(0) {}

  /** Get the SHM serialized size of an argument pack */
  template<typename ...Args>
  HSHM_ALWAYS_INLINE static size_t shm_buf_size(Args&& ...args) {
    size_t size = 0;
    auto lambda = [&size](auto i, auto &&arg) {
      if constexpr(std::is_pod<std::remove_reference<decltype(arg)>>()) {
        size += sizeof(arg);
      } else if constexpr(IS_SHM_ARCHIVEABLE(decltype(arg))) {
        size += sizeof(hipc::OffsetPointer);
      } else {
        throw IPC_ARGS_NOT_SHM_COMPATIBLE.format();
      }
    };
    ForwardIterateArgpack::Apply(make_argpack(std::forward<Args>(args)...), lambda);
    return size;
  }

  /** Serialize a set of arguments into shared memory */
  template<typename ...Args>
  HSHM_ALWAYS_INLINE char* serialize(Allocator *alloc, Args&& ...args) {
    size_t buf_size = sizeof(allocator_id_t) + shm_buf_size(std::forward<Args>(args)...);
    Pointer p;
    char *buf = alloc->AllocatePtr<char>(buf_size, p);
    memcpy(buf, &p.allocator_id_, sizeof(allocator_id_t));
    off_ = sizeof(allocator_id_t);
    auto lambda = [buf, this](auto i, auto &&arg) {
      if constexpr(std::is_pod<std::remove_reference<decltype(arg)>>()) {
        memcpy(buf + this->off_, &arg, sizeof(arg));
        this->off_ += sizeof(arg);
      } else if constexpr(IS_SHM_ARCHIVEABLE(decltype(arg))) {
        OffsetPointer p = arg.ToOffsetPointer();
        memcpy(buf + this->off_, (void*)&p, sizeof(p));
        this->off_ += sizeof(p);
      } else {
        throw IPC_ARGS_NOT_SHM_COMPATIBLE.format();
      }
    };
    ForwardIterateArgpack::Apply(make_argpack(std::forward<Args>(args)...), lambda);
    return buf;
  }

  /** Deserialize an allocator from the SHM buffer */
  HSHM_ALWAYS_INLINE Allocator* deserialize(char *buf) {
    allocator_id_t alloc_id;
    memcpy((void*)&alloc_id, buf + off_, sizeof(allocator_id_t));
    off_ += sizeof(allocator_id_t);
    return HERMES_MEMORY_MANAGER->GetAllocator(alloc_id);
  }

  /** Deserialize an argument from the SHM buffer */
  template<typename T, typename ...Args>
  HSHM_ALWAYS_INLINE T deserialize(Allocator *alloc, char *buf) {
    T arg;
    if constexpr(std::is_pod<T>()) {
      memcpy(&arg, buf + off_, sizeof(arg));
      off_ += sizeof(arg);
    } else if constexpr(IS_SHM_ARCHIVEABLE(T)) {
      OffsetPointer p;
      memcpy((void*)&p, buf + off_, sizeof(p));
      arg.shm_deserialize(alloc, p);
      off_ += sizeof(p);
    } else {
      throw IPC_ARGS_NOT_SHM_COMPATIBLE.format();
    }
    return arg;
  }
};

}  // namespace hshm

#endif  // HERMES_SHM_INCLUDE_HERMES_SHM_DATA_STRUCTURES_SERIALIZATION_SHM_SERIALIZE_H_
