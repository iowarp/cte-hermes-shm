//
// Created by llogan on 11/27/24.
//

#ifndef HERMES_SHM_INCLUDE_HERMES_SHM_DATA_STRUCTURES_SERIALIZATION_LOCAL_SERIALIZE_H_
#define HERMES_SHM_INCLUDE_HERMES_SHM_DATA_STRUCTURES_SERIALIZATION_LOCAL_SERIALIZE_H_


#include "hermes_shm/data_structures/data_structure.h"
#include "serialize_common.h"

namespace hshm::ipc {

template<typename Ar, typename T>
void save(Ar &ar, const cereal::BinaryData<T> &data) {
  ar.write_binary((const char*)data.data, data.size);
}

template<typename Ar, typename T>
void load(Ar &ar, cereal::BinaryData<T> &data) {
  ar.read_binary((char*)data.data, data.size);
}

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
    return base(obj);
  }

  /** Parenthesis operator */
  template<typename ...Args>
  HSHM_ALWAYS_INLINE
  LocalSerialize& operator()(Args&& ...args) {
    hshm::ForwardIterateArgpack::Apply(
        hshm::make_argpack(std::forward<Args>(args)...),
        [this](auto i, auto &arg) {
          this->base(arg);
        });
    return *this;
  }

  /** Save function */
  template<typename T>
  HSHM_ALWAYS_INLINE
  LocalSerialize& base(const T &obj) {
    static_assert(
        is_serializeable_v<LocalSerialize, T>,
        "Cannot serialize object");
    if constexpr (std::is_arithmetic<T>::value) {
      write_binary(reinterpret_cast<const char*>(&obj), sizeof(T));
    } else if constexpr (has_serialize_fun_v<LocalSerialize, T>) {
      serialize(*this, const_cast<T&>(obj));
    } else if constexpr (has_load_save_fun_v<LocalSerialize, T>) {
      save(*this, obj);
    } else if constexpr (has_serialize_cls_v<LocalSerialize, T>) {
      const_cast<T&>(obj).serialize(*this);
    } else if constexpr (has_load_save_cls_v<LocalSerialize, T>) {
      obj.save(*this);
    }
    return *this;
  }

  /** Save function (binary data) */
  HSHM_ALWAYS_INLINE
  LocalSerialize& write_binary(const char *data, size_t size) {
    size_t off = data_.size();
    data_.resize(off + size);
    memcpy(data_.data() + off, data, size);
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
    return base(obj);
  }

  /** Parenthesis operator */
  template<typename ...Args>
  HSHM_ALWAYS_INLINE
  LocalDeserialize& operator()(Args&& ...args) {
    hshm::ForwardIterateArgpack::Apply(
        hshm::make_argpack(std::forward<Args>(args)...),
        [this](auto i, auto &arg) {
          this->base(arg);
        });
    return *this;
  }

  /** Load function */
  template<typename T>
  HSHM_ALWAYS_INLINE
  LocalDeserialize& base(T &obj) {
    static_assert(
        is_serializeable_v<LocalDeserialize, T>,
        "Cannot serialize object");
    if constexpr (std::is_arithmetic<T>::value) {
      read_binary(reinterpret_cast<char*>(&obj), sizeof(T));
    } else if constexpr (has_serialize_fun_v<LocalDeserialize, T>) {
      serialize(*this, obj);
    } else if constexpr (has_load_save_fun_v<LocalDeserialize, T>) {
      load(*this, obj);
    } else if constexpr (has_serialize_cls_v<LocalDeserialize, T>) {
      obj.serialize(*this);
    } else if constexpr (has_load_save_cls_v<LocalDeserialize, T>) {
      obj.load(*this);
    }
    return *this;
  }

  /** Save function (binary data) */
  HSHM_ALWAYS_INLINE
  LocalDeserialize& read_binary(char *data, size_t size) {
    memcpy(data, data_.data() + cur_off_, size);
    cur_off_ += size;
    return *this;
  }
};

}  // namespace chi


#endif  // HERMES_SHM_INCLUDE_HERMES_SHM_DATA_STRUCTURES_SERIALIZATION_LOCAL_SERIALIZE_H_
