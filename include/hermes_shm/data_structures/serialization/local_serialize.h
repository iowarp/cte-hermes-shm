//
// Created by llogan on 11/27/24.
//

#ifndef HERMES_SHM_INCLUDE_HERMES_SHM_DATA_STRUCTURES_SERIALIZATION_LOCAL_SERIALIZE_H_
#define HERMES_SHM_INCLUDE_HERMES_SHM_DATA_STRUCTURES_SERIALIZATION_LOCAL_SERIALIZE_H_


#include "hermes_shm/data_structures/data_structure.h"
#include "serialize_common.h"

namespace hshm::ipc {

/** A class for serializing simple objects into private memory */
template<typename DataT = hshm::charbuf>
class LocalSerialize {
 public:
  DataT &data_;
 public:
  LocalSerialize(DataT &data) : data_(data) {
    data_.resize(0);
  }
  LocalSerialize(DataT &data, bool) : data_(data) {}

  /** left shift operator */
  template<typename T>
  HSHM_ALWAYS_INLINE
  LocalSerialize& operator<<(const T &obj) {
    return save(obj);
  }

  /** Parenthesis operator */
  template<typename ...Args>
  HSHM_ALWAYS_INLINE
  LocalSerialize& operator()(Args&& ...args) {
    hshm::ForwardIterateArgpack::Apply(
        hshm::make_argpack(std::forward<Args>(args)...),
        [this](auto i, const auto &arg) {
          this->save(arg);
        });
    return *this;
  }

  /** Save function */
  template<typename T>
  HSHM_ALWAYS_INLINE
  LocalSerialize& save(const T &obj) {
    if constexpr(std::is_arithmetic<T>::value) {
      size_t size = sizeof(T);
      size_t off = data_.size();
      data_.resize(off + size);
      memcpy(data_.data() + off, &obj, size);
    } else if constexpr (has_serialize_fun_v<LocalSerialize, T>) {
      serialize(*this, const_cast<T&>(obj));
    } else if constexpr (has_load_fun_save_fun_v<LocalSerialize, T>) {
      save(*this, obj);
    } else if constexpr (has_serialize_cls_v<LocalSerialize, T>) {
      const_cast<T&>(obj).serialize(*this);
    } else if constexpr (has_save_cls_v<LocalSerialize, T>) {
      obj.save(*this);
    } else {
      static_assert("Cannot serialize object");
    }
    return *this;
  }
};

/** A class for serializing simple objects into private memory */
template<typename DataT = hshm::charbuf>
class LocalDeserialize {
 public:
  const DataT &data_;
  size_t cur_off_ = 0;
 public:
  LocalDeserialize(const DataT &data) : data_(data) {
    cur_off_ = 0;
  }

  /** right shift operator */
  template<typename T>
  HSHM_ALWAYS_INLINE
  LocalDeserialize& operator>>(T &obj) {
    return load(obj);
  }

  /** Parenthesis operator */
  template<typename ...Args>
  HSHM_ALWAYS_INLINE
  LocalDeserialize& operator()(Args&& ...args) {
    hshm::ForwardIterateArgpack::Apply(
        hshm::make_argpack(std::forward<Args>(args)...),
        [this](auto i, auto &arg) {
          this->load(arg);
        });
    return *this;
  }

  /** Load function */
  template<typename T>
  HSHM_ALWAYS_INLINE
  LocalDeserialize& load(T &obj) {
    if constexpr(std::is_arithmetic<T>::value) {
      size_t size = sizeof(T);
      memcpy(&obj, data_.data() + cur_off_, size);
      cur_off_ += size;
    } else if constexpr (has_serialize_fun_v<LocalDeserialize, T>) {
      serialize(*this, obj);
    } else if constexpr (has_load_fun_save_fun_v<LocalDeserialize, T>) {
      load(*this, obj);
    } else if constexpr (has_serialize_cls_v<LocalDeserialize, T>) {
      obj.serialize(*this);
    } else if constexpr (has_load_cls_v<LocalDeserialize, T>) {
      obj.load(*this);
    } else {
      static_assert("Cannot serialize object");
    }
    return *this;
  }
};

}  // namespace chi


#endif  // HERMES_SHM_INCLUDE_HERMES_SHM_DATA_STRUCTURES_SERIALIZATION_LOCAL_SERIALIZE_H_
