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

#ifndef HSHM_DATA_STRUCTURES_UNORDERED_MAP_H_
#define HSHM_DATA_STRUCTURES_UNORDERED_MAP_H_

#include "hermes_shm/data_structures/internal/shm_internal.h"
#include "hermes_shm/data_structures/ipc/slist.h"
#include "hermes_shm/data_structures/ipc/vector.h"
#include "hermes_shm/thread/thread_model_manager.h"
#include "hermes_shm/types/atomic.h"
#include "pair.h"

namespace hshm::ipc {

/** forward pointer for unordered_map */
template <typename Key, typename T, class Hash = hshm::hash<Key>,
          HSHM_CLASS_TEMPL_WITH_DEFAULTS>
class unordered_map;

/**
 * The unordered map iterator (bucket_iter, slist_iter)
 * */
template <typename Key, typename T, class Hash, HSHM_CLASS_TEMPL>
struct unordered_map_iterator {
 public:
  using COLLISION_T = hipc::pair<Key, T, HSHM_CLASS_TEMPL_ARGS>;
  using BUCKET_T = hipc::slist<COLLISION_T, HSHM_CLASS_TEMPL_ARGS>;
  using BUCKET_VEC_T = hipc::vector<BUCKET_T, HSHM_CLASS_TEMPL_ARGS>;
  using COLLISION_LIST_T = hipc::slist<COLLISION_T, HSHM_CLASS_TEMPL_ARGS>;

 public:
  unordered_map<Key, T, Hash, HSHM_CLASS_TEMPL_ARGS> *map_;
  typename BUCKET_VEC_T::iterator_t bucket_;
  typename COLLISION_LIST_T::iterator_t collision_;

  /** Default constructor */
  HSHM_CROSS_FUN unordered_map_iterator() = default;

  /** Construct the iterator  */
  HSHM_INLINE_CROSS_FUN explicit unordered_map_iterator(
      unordered_map<Key, T, Hash, HSHM_CLASS_TEMPL_ARGS> &map)
      : map_(&map) {}

  /** Copy constructor  */
  HSHM_INLINE_CROSS_FUN unordered_map_iterator(
      const unordered_map_iterator &other) {
    shm_strong_copy(other);
  }

  /** Assign one iterator into another */
  HSHM_INLINE_CROSS_FUN unordered_map_iterator &operator=(
      const unordered_map_iterator &other) {
    if (this != &other) {
      shm_strong_copy(other);
    }
    return *this;
  }

  /** Copy an iterator */
  HSHM_CROSS_FUN
  void shm_strong_copy(const unordered_map_iterator &other) {
    map_ = other.map_;
    bucket_ = other.bucket_;
    collision_ = other.collision_;
  }

  /** Get the pointed object */
  HSHM_INLINE_CROSS_FUN COLLISION_T &operator*() { return *collision_; }

  /** Get the pointed object */
  HSHM_INLINE_CROSS_FUN const COLLISION_T &operator*() const {
    return *collision_;
  }

  /** Go to the next object */
  HSHM_INLINE_CROSS_FUN unordered_map_iterator &operator++() {
    ++collision_;
    make_correct();
    return *this;
  }

  /** Return the next iterator */
  HSHM_INLINE_CROSS_FUN unordered_map_iterator operator++(int) const {
    unordered_map_iterator next(*this);
    ++next;
    return next;
  }

  /**
   * Shifts bucket and collision iterator until there is a valid element.
   * Returns true if such an element is found, and false otherwise.
   * */
  HSHM_INLINE_CROSS_FUN bool make_correct() {
    do {
      if (bucket_.is_end()) {
        return false;
      }
      if (!collision_.is_end()) {
        return true;
      } else {
        ++bucket_;
        if (bucket_.is_end()) {
          return false;
        }
        BUCKET_T &bkt = *bucket_;
        collision_ = bkt.begin();
      }
    } while (true);
  }

  /** Check if two iterators are equal */
  HSHM_INLINE_CROSS_FUN friend bool operator==(
      const unordered_map_iterator &a, const unordered_map_iterator &b) {
    if (a.is_end() && b.is_end()) {
      return true;
    }
    return (a.bucket_ == b.bucket_) && (a.collision_ == b.collision_);
  }

  /** Check if two iterators are inequal */
  HSHM_INLINE_CROSS_FUN friend bool operator!=(
      const unordered_map_iterator &a, const unordered_map_iterator &b) {
    if (a.is_end() && b.is_end()) {
      return false;
    }
    return (a.bucket_ != b.bucket_) || (a.collision_ != b.collision_);
  }

  /** Determine whether this iterator is the end iterator */
  HSHM_INLINE_CROSS_FUN bool is_end() const { return bucket_.is_end(); }

  /** Set this iterator to the end iterator */
  HSHM_INLINE_CROSS_FUN void set_end() { bucket_.set_end(); }
};

/**
 * MACROS to simplify the unordered_map namespace
 * Used as inputs to the HIPC_CONTAINER_TEMPLATE
 * */

#define CLASS_NAME unordered_map
#define CLASS_NEW_ARGS Key, T, Hash

/**
 * The unordered map implementation
 * */
template <typename Key, typename T, class Hash, HSHM_CLASS_TEMPL>
class unordered_map : public ShmContainer {
 public:
  HIPC_CONTAINER_TEMPLATE((CLASS_NAME), (CLASS_NEW_ARGS))

  /**====================================
   * Typedefs
   * ===================================*/
  typedef unordered_map_iterator<Key, T, Hash, HSHM_CLASS_TEMPL_ARGS>
      iterator_t;
  friend iterator_t;
  using COLLISION_T = hipc::pair<Key, T, HSHM_CLASS_TEMPL_ARGS>;
  using BUCKET_T = hipc::slist<COLLISION_T, HSHM_CLASS_TEMPL_ARGS>;
  using BUCKET_VEC_T = hipc::vector<BUCKET_T, HSHM_CLASS_TEMPL_ARGS>;

  /**====================================
   * Variables
   * ===================================*/
  delay_ar<BUCKET_VEC_T> buckets_;
  RealNumber max_capacity_;
  RealNumber growth_;
  hipc::nonatomic<hshm::size_t> length_;

 public:
  /**====================================
   * Default Constructor
   * ===================================*/

  /**
   * SHM constructor. Initialize the map.
   *
   * @param num_buckets the number of buckets to create
   * @param max_capacity the maximum number of elements before a growth is
   * triggered
   * @param growth the multiplier to grow the bucket vector size
   * */
  HSHM_CROSS_FUN
  explicit unordered_map(int num_buckets = 20,
                         RealNumber max_capacity = RealNumber(4, 5),
                         RealNumber growth = RealNumber(5, 4)) {
    shm_init(HSHM_MEMORY_MANAGER->GetDefaultAllocator<AllocT>(), num_buckets,
             max_capacity, growth);
  }

  /**
   * SHM constructor. Initialize the map.
   *
   * @param alloc the shared-memory allocator
   * @param num_buckets the number of buckets to create
   * @param max_capacity the maximum number of elements before a growth is
   * triggered
   * @param growth the multiplier to grow the bucket vector size
   * */
  HSHM_CROSS_FUN
  explicit unordered_map(const hipc::CtxAllocator<AllocT> &alloc,
                         int num_buckets = 20,
                         RealNumber max_capacity = RealNumber(4, 5),
                         RealNumber growth = RealNumber(5, 4)) {
    shm_init(alloc, num_buckets, max_capacity, growth);
  }

  /** SHM constructor. */
  HSHM_CROSS_FUN
  void shm_init(const hipc::CtxAllocator<AllocT> &alloc, int num_buckets = 20,
                RealNumber max_capacity = RealNumber(4, 5),
                RealNumber growth = RealNumber(5, 4)) {
    init_shm_container(alloc);
    buckets_.shm_init(GetCtxAllocator(), num_buckets);
    max_capacity_ = max_capacity;
    growth_ = growth;
    length_ = 0;
  }

  /**====================================
   * Copy Constructors
   * ===================================*/

  /** Copy constructor */
  HSHM_CROSS_FUN
  explicit unordered_map(const unordered_map &other) {
    init_shm_container(HSHM_MEMORY_MANAGER->GetDefaultAllocator<AllocT>());
    shm_strong_copy_construct(other);
  }

  /** SHM copy constructor */
  HSHM_CROSS_FUN
  explicit unordered_map(const hipc::CtxAllocator<AllocT> &alloc,
                         const unordered_map &other) {
    init_shm_container(alloc);
    shm_strong_copy_construct(other);
  }

  /** SHM copy constructor main */
  HSHM_CROSS_FUN
  void shm_strong_copy_construct(const unordered_map &other) {
    SetNull();
    buckets_.shm_init(GetCtxAllocator(), other.GetBuckets());
    shm_strong_copy_op(other);
  }

  /** SHM copy assignment operator */
  HSHM_CROSS_FUN
  unordered_map &operator=(const unordered_map &other) {
    if (this != &other) {
      shm_destroy();
      shm_strong_copy_op(other);
    }
    return *this;
  }

  /** Internal copy operation */
  HSHM_CROSS_FUN
  void shm_strong_copy_op(const unordered_map &other) {
    int num_buckets = other.get_num_buckets();
    GetBuckets().resize(num_buckets);
    max_capacity_ = other.max_capacity_;
    growth_ = other.growth_;
    for (hipc::pair<Key, T, HSHM_CLASS_TEMPL_ARGS> &entry : other) {
      emplace_templ<false, true>(entry.GetKey(), entry.GetVal());
    }
  }

  /**====================================
   * Move Constructors
   * ===================================*/

  /** Move constructor. */
  HSHM_INLINE_CROSS_FUN unordered_map(unordered_map &&other) noexcept {
    shm_move_op<false>(other.GetCtxAllocator(), std::move(other));
  }

  /** SHM move constructor. */
  HSHM_INLINE_CROSS_FUN unordered_map(const hipc::CtxAllocator<AllocT> &alloc,
                                      unordered_map &&other) noexcept {
    shm_move_op<false>(alloc, std::move(other));
  }

  /** SHM move assignment operator. */
  HSHM_CROSS_FUN
  unordered_map &operator=(unordered_map &&other) noexcept {
    if (this != &other) {
      shm_move_op<true>(GetCtxAllocator(), std::move(other));
    }
    return *this;
  }

  /** SHM move operator. */
  template <bool IS_ASSIGN>
  HSHM_CROSS_FUN void shm_move_op(const hipc::CtxAllocator<AllocT> &alloc,
                                  unordered_map &&other) noexcept {
    if constexpr (!IS_ASSIGN) {
      init_shm_container(alloc);
    }
    if (GetAllocator() == other.GetAllocator()) {
      if constexpr (IS_ASSIGN) {
        GetBuckets() = std::move(other.GetBuckets());
      } else {
        buckets_.shm_init(GetCtxAllocator(),
                           std::move(other.GetBuckets()));
      }
      max_capacity_ = other.max_capacity_;
      growth_ = other.growth_;
      length_ = other.length_.load();
      other.SetNull();
    } else {
      shm_strong_copy_op(other);
      other.shm_destroy();
    }
  }

  /**====================================
   * Destructor
   * ===================================*/

  /** Check if the pair is empty */
  HSHM_INLINE_CROSS_FUN bool IsNull() { return buckets_->IsNull(); }

  /** Sets this pair as empty */
  HSHM_INLINE_CROSS_FUN void SetNull() { buckets_->SetNull(); }

  /** Destroy the unordered_map buckets */
  HSHM_INLINE_CROSS_FUN void shm_destroy_main() {
    BUCKET_VEC_T &buckets = GetBuckets();
    buckets.shm_destroy();
  }

  /**====================================
   * Emplace Methods
   * ===================================*/

  /**
   * Construct an object directly in the map. Overrides the object if
   * key already exists.
   *
   * @param key the key to future index the map
   * @param args the arguments to construct the object
   * @return None
   * */
  template <typename... Args>
  HSHM_CROSS_FUN bool emplace(const Key &key, Args &&...args) {
    return emplace_templ<true, true>(key, std::forward<Args>(args)...);
  }

  /**
   * Construct an object directly in the map. Does not modify the key
   * if it already exists.
   *
   * @param key the key to future index the map
   * @param args the arguments to construct the object
   * @return None
   * */
  template <typename... Args>
  HSHM_CROSS_FUN bool try_emplace(const Key &key, Args &&...args) {
    return emplace_templ<true, false>(key, std::forward<Args>(args)...);
  }

 private:
  /**
   * Insert a serialized (key, value) pair in the map
   *
   * @param growth whether or not to grow the unordered map on collision
   * @param modify_existing whether or not to override an existing entry
   * @param entry the (key,value) pair shared-memory serialized
   * @return None
   * */
  template <bool growth, bool modify_existing, typename... Args>
  HSHM_INLINE_CROSS_FUN bool emplace_templ(const Key &key, Args &&...args) {
    // Hash the key to a bucket
    BUCKET_VEC_T &buckets = GetBuckets();
    size_t bkt_id = Hash{}(key) % buckets.size();
    BUCKET_T &bkt = (buckets)[bkt_id];

    // Insert into the map
    auto has_key_iter = find_collision(key, bkt);
    if (!has_key_iter.is_end()) {
      if constexpr (!modify_existing) {
        return false;
      } else {
        bkt.erase(has_key_iter);
        --length_;
      }
    }
    bkt.emplace_back(PiecewiseConstruct(), make_argpack(key),
                     make_argpack(std::forward<Args>(args)...));

    // Increment the size of the map
    ++length_;
    return true;
  }

 public:
  /**====================================
   * Erase Methods
   * ===================================*/

  /**
   * Erase an object indexable by \a key key
   * */
  HSHM_CROSS_FUN
  void erase(const Key &key) {
    // Get the bucket the key belongs to
    BUCKET_VEC_T &buckets = GetBuckets();
    size_t bkt_id = Hash{}(key) % buckets.size();
    BUCKET_T &bkt = (buckets)[bkt_id];

    // Find and remove key from collision slist
    auto iter = find_collision(key, bkt);
    if (iter.is_end()) {
      return;
    }
    bkt.erase(iter);

    // Decrement the size of the map
    --length_;
  }

  /**
   * Erase an object at the iterator
   * */
  HSHM_CROSS_FUN
  void erase(iterator_t &iter) {
    if (iter == end()) return;
    // Acquire the bucket lock for a write (modifying collisions)
    BUCKET_T &bkt = *iter.bucket_;

    // Erase the element from the collision slist
    bkt.erase(iter.collision_);

    // Decrement the size of the map
    --length_;
  }

  /**
   * Erase the entire map
   * */
  HSHM_CROSS_FUN void clear() {
    BUCKET_VEC_T &buckets = GetBuckets();
    size_t num_buckets = buckets.size();
    buckets.clear();
    buckets.resize(num_buckets);
    length_ = 0;
  }

  /**====================================
   * Index Methods
   * ===================================*/

  /**
   * Locate an entry in the unordered_map
   *
   * @return the object pointed by key
   * @exception UNORDERED_MAP_CANT_FIND the key was not in the map
   * */
  HSHM_INLINE_CROSS_FUN T &operator[](const Key &key) {
    auto iter = find(key);
    if (!iter.is_end()) {
      return (*iter).second_.get_ref();
    }
    HSHM_THROW_ERROR(UNORDERED_MAP_CANT_FIND);
  }

  /** Find an object in the unordered_map */
  HSHM_CROSS_FUN
  iterator_t find(const Key &key) {
    iterator_t iter(*this);

    // Determine the bucket corresponding to the key
    BUCKET_VEC_T &buckets = GetBuckets();
    size_t bkt_id = Hash{}(key) % buckets.size();
    iter.bucket_ = buckets.begin() + bkt_id;
    BUCKET_T &bkt = (*iter.bucket_);

    // Get the specific collision iterator
    iter.collision_ = find_collision(key, bkt);
    if (iter.collision_.is_end()) {
      iter.set_end();
    }
    return iter;
  }

  /** Find a key in the collision slist */
  typename BUCKET_T::iterator_t HSHM_INLINE_CROSS_FUN
  find_collision(const Key &key, BUCKET_T &bkt) {
    auto iter = bkt.begin();
    auto iter_end = bkt.end();
    for (; iter != iter_end; ++iter) {
      COLLISION_T &collision = *iter;
      if (collision.GetKey() == key) {
        return iter;
      }
    }
    return iter_end;
  }

  /**====================================
   * Query Methods
   * ===================================*/

  /** The number of entries in the map */
  HSHM_INLINE_CROSS_FUN size_t size() const { return (size_t)length_.load(); }

  /** The number of buckets in the map */
  HSHM_INLINE_CROSS_FUN size_t get_num_buckets() const {
    BUCKET_VEC_T &buckets = GetBuckets();
    return buckets.size();
  }

 public:
  /**====================================
   * Iterators
   * ===================================*/

  /** Forward iterator begin */
  HSHM_INLINE_CROSS_FUN iterator_t begin() const {
    iterator_t iter(const_cast<unordered_map &>(*this));
    BUCKET_VEC_T &buckets(GetBuckets());
    if (buckets.size() == 0) {
      return iter;
    }
    BUCKET_T &bkt = buckets[0];
    iter.bucket_ = buckets.cbegin();
    iter.collision_ = bkt.begin();
    iter.make_correct();
    return iter;
  }

  /** Forward iterator end */
  HSHM_INLINE_CROSS_FUN iterator_t end() const {
    iterator_t iter(const_cast<unordered_map &>(*this));
    BUCKET_VEC_T &buckets(GetBuckets());
    iter.bucket_ = buckets.cend();
    return iter;
  }

  /** Get the buckets */
  HSHM_INLINE_CROSS_FUN BUCKET_VEC_T &GetBuckets() { return *buckets_; }

  /** Get the buckets (const) */
  HSHM_INLINE_CROSS_FUN BUCKET_VEC_T &GetBuckets() const {
    return const_cast<BUCKET_VEC_T &>(*buckets_);
  }
};

}  // namespace hshm::ipc

namespace hshm {

template <typename Key, typename T, class Hash = hshm::hash<Key>,
          HSHM_CLASS_TEMPL_WITH_PRIV_DEFAULTS>
using unordered_map = hipc::unordered_map<Key, T, Hash, HSHM_CLASS_TEMPL_ARGS>;

}  // namespace hshm

#undef CLASS_NAME
#undef CLASS_NEW_ARGS

#endif  // HSHM_DATA_STRUCTURES_UNORDERED_MAP_H_
