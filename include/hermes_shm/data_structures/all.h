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

#ifndef HSHM_DATA_STRUCTURES_DATA_STRUCTURE_H_
#define HSHM_DATA_STRUCTURES_DATA_STRUCTURE_H_

#include "hermes_shm/memory/memory_manager.h"
#include "internal/shm_internal.h"
#include "ipc/charwrap.h"
#include "ipc/chararr.h"
#include "ipc/dynamic_queue.h"
#include "ipc/functional.h"
#include "ipc/key_set.h"
#include "ipc/lifo_list_queue.h"
#include "ipc/list.h"
#include "ipc/mpsc_lifo_list_queue.h"
#include "ipc/multi_ring_buffer.h"
#include "ipc/pair.h"
#include "ipc/ring_ptr_queue.h"
#include "ipc/ring_queue.h"
#include "ipc/slist.h"
#include "ipc/split_ticket_queue.h"
#include "ipc/spsc_fifo_list_queue.h"
#include "ipc/string.h"
#include "ipc/ticket_queue.h"
#include "ipc/tuple_base.h"
#include "ipc/unordered_map.h"
#include "ipc/vector.h"
#include "serialization/local_serialize.h"
#include "serialization/serialize_common.h"

#define HSHM_DATA_STRUCTURES_TEMPLATE_BASE(NS, HSHM_NS, ALLOC_T)             \
  namespace NS {                                                             \
  template <int LENGTH, bool WithNull>                                       \
  using chararr_templ = HSHM_NS::chararr_templ<LENGTH, WithNull>;            \
                                                                             \
  using HSHM_NS::chararr;                                                    \
                                                                             \
  template <typename T>                                                      \
  using lifo_list_queue = HSHM_NS::lifo_list_queue<T, ALLOC_T>;              \
                                                                             \
  template <typename T>                                                      \
  using list = HSHM_NS::list<T, ALLOC_T>;                                    \
                                                                             \
  template <typename T>                                                      \
  using mpsc_lifo_list_queue = HSHM_NS::mpsc_lifo_list_queue<T, ALLOC_T>;    \
                                                                             \
  template <typename T>                                                      \
  using spsc_fifo_list_queue = HSHM_NS::spsc_fifo_list_queue<T, ALLOC_T>;    \
                                                                             \
  template <typename FirstT, typename SecondT>                               \
  using pair = HSHM_NS::pair<FirstT, SecondT, ALLOC_T>;                      \
                                                                             \
  template <typename T, typename HDR = hshm::ipc::EmptyHeader>                 \
  using spsc_queue = HSHM_NS::spsc_queue<T, HDR, ALLOC_T>;                   \
  template <typename T, typename HDR = hshm::ipc::EmptyHeader>                 \
  using mpsc_queue = HSHM_NS::mpsc_queue<T, HDR, ALLOC_T>;                   \
  template <typename T, typename HDR = hshm::ipc::EmptyHeader>                 \
  using fixed_spsc_queue = HSHM_NS::fixed_spsc_queue<T, HDR, ALLOC_T>;       \
  template <typename T, typename HDR = hshm::ipc::EmptyHeader>                 \
  using fixed_mpsc_queue = HSHM_NS::fixed_mpsc_queue<T, HDR, ALLOC_T>;       \
  template <typename T, typename HDR = hshm::ipc::EmptyHeader>                 \
  using circular_mpsc_queue = HSHM_NS::circular_mpsc_queue<T, HDR, ALLOC_T>; \
  template <typename T, typename HDR = hshm::ipc::EmptyHeader>                 \
  using circular_spsc_queue = HSHM_NS::circular_spsc_queue<T, HDR, ALLOC_T>; \
  template <typename T, typename HDR = hshm::ipc::EmptyHeader>                 \
  using ext_ring_buffer = HSHM_NS::ext_ring_buffer<T, HDR, ALLOC_T>;              \
                                                                             \
  template <typename T, typename HDR = hshm::ipc::EmptyHeader>                 \
  using multi_mpsc_queue = HSHM_NS::multi_mpsc_queue<T, HDR, ALLOC_T>;       \
  template <typename T, typename HDR = hshm::ipc::EmptyHeader>                 \
  using multi_spsc_queue = HSHM_NS::multi_spsc_queue<T, HDR, ALLOC_T>;       \
  template <typename T, typename HDR = hshm::ipc::EmptyHeader>                 \
  using multi_fixed_mpsc_queue = HSHM_NS::multi_fixed_mpsc_queue<T, HDR, ALLOC_T>;\
  template <typename T, typename HDR = hshm::ipc::EmptyHeader>                 \
  using multi_circular_mpsc_queue =                                          \
      HSHM_NS::multi_circular_mpsc_queue<T, HDR, ALLOC_T>;                        \
                                                                             \
  template <typename T, typename HDR = hshm::ipc::EmptyHeader>                 \
  using spsc_ptr_queue = HSHM_NS::spsc_ptr_queue<T, HDR, ALLOC_T>;           \
  template <typename T, typename HDR = hshm::ipc::EmptyHeader>                 \
  using mpsc_ptr_queue = HSHM_NS::mpsc_ptr_queue<T, HDR, ALLOC_T>;           \
  template <typename T, typename HDR = hshm::ipc::EmptyHeader>                 \
  using fixed_spsc_ptr_queue = HSHM_NS::fixed_spsc_ptr_queue<T, HDR, ALLOC_T>;\
  template <typename T, typename HDR = hshm::ipc::EmptyHeader>                 \
  using fixed_mpsc_ptr_queue = HSHM_NS::fixed_mpsc_ptr_queue<T, HDR, ALLOC_T>;\
  template <typename T, typename HDR = hshm::ipc::EmptyHeader>                 \
  using circular_mpsc_ptr_queue =                                            \
      HSHM_NS::circular_mpsc_ptr_queue<T, HDR, ALLOC_T>;                     \
  template <typename T, typename HDR = hshm::ipc::EmptyHeader>                 \
  using circular_spsc_ptr_queue = HSHM_NS::circular_spsc_ptr_queue<T, HDR, ALLOC_T>;  \
  template <typename T, typename HDR = hshm::ipc::EmptyHeader>                 \
  using ext_ptr_ring_buffer = HSHM_NS::ext_ptr_ring_buffer<T, HDR, ALLOC_T>;          \
                                                                             \
  template <typename T>                                                      \
  using slist = HSHM_NS::slist<T, ALLOC_T>;                                  \
                                                                             \
  template <typename T>                                                      \
  using split_ticket_queue = HSHM_NS::split_ticket_queue<T, ALLOC_T>;        \
                                                                             \
  using string = HSHM_NS::string_templ<HSHM_STRING_SSO, 0, ALLOC_T>;         \
                                                                             \
  using charbuf = HSHM_NS::string_templ<HSHM_STRING_SSO, 0, ALLOC_T>;        \
                                                                             \
  using charwrap = HSHM_NS::string_templ<HSHM_STRING_SSO,                    \
                                         hipc::StringFlags::kWrap, ALLOC_T>; \
                                                                             \
  template <typename T>                                                      \
  using ticket_queue = HSHM_NS::ticket_queue<T, ALLOC_T>;                    \
                                                                             \
  template <typename Key, typename T, class Hash = hshm::hash<Key>>          \
  using unordered_map = HSHM_NS::unordered_map<Key, T, Hash, ALLOC_T>;       \
                                                                             \
  template <typename T>                                                      \
  using vector = HSHM_NS::vector<T, ALLOC_T>;                                \
                                                                             \
  template <typename T>                                                      \
  using spsc_key_set = HSHM_NS::spsc_key_set<T, ALLOC_T>;                    \
                                                                             \
  template <typename T>                                                      \
  using mpmc_key_set = HSHM_NS::mpmc_key_set<T, ALLOC_T>;                    \
                                                                             \
  template <typename T>                                                      \
  using dynamic_queue = HSHM_NS::dynamic_queue<T, ALLOC_T>;                  \
  }  // namespace NS

#define HSHM_DATA_STRUCTURES_TEMPLATE(NS, ALLOC_T)      \
  HSHM_DATA_STRUCTURES_TEMPLATE_BASE(NS, hshm, ALLOC_T) \
  HSHM_DATA_STRUCTURES_TEMPLATE_BASE(NS::ipc, hipc, ALLOC_T)

#endif  // HSHM_DATA_STRUCTURES_DATA_STRUCTURE_H_
