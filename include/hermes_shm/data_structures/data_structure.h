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


#ifndef HERMES_DATA_STRUCTURES_DATA_STRUCTURE_H_
#define HERMES_DATA_STRUCTURES_DATA_STRUCTURE_H_

#include "ipc/internal/shm_internal.h"
#include "hermes_shm/memory/memory_manager.h"

#include "containers/charbuf.h"
#include "containers/functional.h"
#include "containers/tuple_base.h"

#include "ipc/iqueue.h"
#include "ipc/list.h"
#include "ipc/pair.h"
#include "ipc/pod_array.h"
#include "ipc/ring_ptr_queue.h"
#include "ipc/ring_queue.h"
#include "ipc/slist.h"
#include "ipc/split_ticket_queue.h"
#include "ipc/string.h"
#include "ipc/ticket_queue.h"
#include "ipc/unordered_map.h"
#include "ipc/vector.h"

#include "serialization/serialize_common.h"

#define HSHM_DEFAULT_MEM_CTX {}

#define HSHM_DATA_STRUCTURES_TEMPLATE(NS, AllocT) \
namespace NS { \
template<typename T> \
using iqueue = hipc::iqueue<T, AllocT>; \
 \
template<typename T> \
using list = hipc::list<T, AllocT>; \
 \
template<typename FirstT, typename SecondT> \
using pair = hipc::pair<FirstT, SecondT, AllocT>; \
 \
template<typename T> \
using spsc_queue = hipc::spsc_queue<T, AllocT>; \
template<typename T> \
using mpsc_queue = hipc::mpsc_queue<T, AllocT>; \
template<typename T> \
using fixed_spsc_queue = hipc::fixed_spsc_queue<T, AllocT>; \
template<typename T> \
using fixed_mpsc_queue = hipc::fixed_mpsc_queue<T, AllocT>; \
template<typename T> \
using fixed_mpmc_queue = hipc::fixed_mpmc_queue<T, AllocT>; \
 \
template<typename T> \
using spsc_ptr_queue = hipc::spsc_ptr_queue<T, AllocT>; \
template<typename T> \
using mpsc_ptr_queue = hipc::mpsc_ptr_queue<T, AllocT>; \
template<typename T> \
using fixed_spsc_ptr_queue = hipc::fixed_spsc_ptr_queue<T, AllocT>; \
template<typename T> \
using fixed_mpsc_ptr_queue = hipc::fixed_mpsc_ptr_queue<T, AllocT>; \
template<typename T> \
using fixed_mpmc_ptr_queue = hipc::fixed_mpmc_ptr_queue<T, AllocT>; \
 \
template<typename T> \
using slist = hipc::slist<T, AllocT>; \
 \
template<typename T> \
using split_ticket_queue = hipc::split_ticket_queue<T, AllocT>; \
 \
using string = hipc::string_templ<32, AllocT>; \
using charbuf = hipc::string_templ<32, AllocT>; \
 \
template<typename T> \
using ticket_queue = hipc::ticket_queue<T, AllocT>; \
 \
template<typename Key, typename T, class Hash = hshm::hash<Key>> \
using unordered_map = hipc::unordered_map<Key, T, Hash, AllocT>; \
 \
template<typename T> \
using vector = hipc::vector<T, AllocT>; \
 \
}

#endif  // HERMES_DATA_STRUCTURES_DATA_STRUCTURE_H_
