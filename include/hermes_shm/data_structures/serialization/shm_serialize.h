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

#ifndef HERMES_SHM_DATA_STRUCTURES_SERIALIZATION_SHM_SERIALIZE_H_
#define HERMES_SHM_DATA_STRUCTURES_SERIALIZATION_SHM_SERIALIZE_H_

#define NOREF typename std::remove_reference<decltype(arg)>::type

namespace hshm::ipc {

class ShmSerializer {
 public:
  size_t off_;
  char *buf_;
  Pointer p_;

  /** Default constructor */
  template<typename ...Args>
  HSHM_CROSS_FUN
  ShmSerializer(Allocator *alloc, Args&& ...args) : off_(0) {
    size_t buf_size = sizeof(AllocatorId) + shm_buf_size(
        std::forward<Args>(args)...);
    buf_ = alloc->AllocatePtr<char>(buf_size, p_);
    auto lambda = [this](auto i, auto &&arg) {
      if constexpr(IS_SHM_ARCHIVEABLE(NOREF)) {
        OffsetPointer p = arg.template GetShmPointer<OffsetPointer>();
        memcpy(this->buf_ + this->off_, (void*)&p, sizeof(p));
        this->off_ += sizeof(p);
      } else if constexpr(std::is_pod<NOREF>()) {
        memcpy(this->buf_ + this->off_, &arg, sizeof(arg));
        this->off_ += sizeof(arg);
      } else {
        HERMES_THROW_ERROR(IPC_ARGS_NOT_SHM_COMPATIBLE);
      }
    };
    ForwardIterateArgpack::Apply(make_argpack(
        std::forward<Args>(args)...), lambda);
  }

  /** Get the SHM serialized size of an argument pack */
  template<typename ...Args>
  HSHM_INLINE_CROSS_FUN
  static size_t shm_buf_size(Args&& ...args) {
    size_t size = 0;
    auto lambda = [&size](auto i, auto &&arg) {
      if constexpr(IS_SHM_ARCHIVEABLE(NOREF)) {
        size += sizeof(hipc::OffsetPointer);
      } else if constexpr(std::is_pod<NOREF>()) {
        size += sizeof(arg);
      } else {
        HERMES_THROW_ERROR(IPC_ARGS_NOT_SHM_COMPATIBLE);
      }
    };
    ForwardIterateArgpack::Apply(make_argpack(
      std::forward<Args>(args)...), lambda);
    return size;
  }
};

class ShmDeserializer {
 public:
  size_t off_;

 public:
  /** Default constructor */
  ShmDeserializer() : off_(0) {}

  /** Deserialize an argument from the SHM buffer */
  template<typename T, typename ...Args>
  HSHM_INLINE_CROSS_FUN T deserialize(Allocator *alloc, char *buf) {
    if constexpr(std::is_pod<T>()) {
      T arg;
      memcpy(&arg, buf + off_, sizeof(arg));
      off_ += sizeof(arg);
      return arg;
    } else {
      HERMES_THROW_ERROR(IPC_ARGS_NOT_SHM_COMPATIBLE);
    }
  }

  /** Deserialize an argument from the SHM buffer */
  template<typename T, typename ...Args>
  HSHM_INLINE_CROSS_FUN void deserialize(Allocator *alloc,
                                      char *buf, hipc::mptr<T> &arg) {
    if constexpr(IS_SHM_ARCHIVEABLE(T)) {
      OffsetPointer p;
      memcpy((void*)&p, buf + off_, sizeof(p));
      arg.shm_deserialize(alloc, p);
      off_ += sizeof(p);
    } else {
      HERMES_THROW_ERROR(IPC_ARGS_NOT_SHM_COMPATIBLE);
    }
  }
};

}  // namespace hshm::ipc

#undef NOREF

#endif  // HERMES_SHM_DATA_STRUCTURES_SERIALIZATION_SHM_SERIALIZE_H_
