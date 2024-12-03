//#include "hermes_shm/data_structures/data_structure.h"
//
//typedef HSHM_NS::Allocator AllocT;

namespace NS {
template<typename T>
using iqueue = HSHM_NS::iqueue<T, AllocT>;

template<typename T>
using list = HSHM_NS::list<T, AllocT>;

template<typename FirstT, typename SecondT>
using pair = HSHM_NS::pair<FirstT, SecondT, AllocT>;

template<typename T>
using spsc_queue = HSHM_NS::spsc_queue<T, AllocT>;
template<typename T>
using mpsc_queue = HSHM_NS::mpsc_queue<T, AllocT>;
template<typename T>
using fixed_spsc_queue = HSHM_NS::fixed_spsc_queue<T, AllocT>;
template<typename T>
using fixed_mpsc_queue = HSHM_NS::fixed_mpsc_queue<T, AllocT>;
template<typename T>
using fixed_mpmc_queue = HSHM_NS::fixed_mpmc_queue<T, AllocT>;

template<typename T>
using spsc_ptr_queue = HSHM_NS::spsc_ptr_queue<T, AllocT>;
template<typename T>
using mpsc_ptr_queue = HSHM_NS::mpsc_ptr_queue<T, AllocT>;
template<typename T>
using fixed_spsc_ptr_queue = HSHM_NS::fixed_spsc_ptr_queue<T, AllocT>;
template<typename T>
using fixed_mpsc_ptr_queue = HSHM_NS::fixed_mpsc_ptr_queue<T, AllocT>;
template<typename T>
using fixed_mpmc_ptr_queue = HSHM_NS::fixed_mpmc_ptr_queue<T, AllocT>;

template<typename T>
using slist = HSHM_NS::slist<T, AllocT>;

template<typename T>
using split_ticket_queue = HSHM_NS::split_ticket_queue<T, AllocT>;

using string = HSHM_NS::string_templ<32, AllocT>;
using charbuf = HSHM_NS::string_templ<32, AllocT>;

template<typename T>
using ticket_queue = HSHM_NS::ticket_queue<T, AllocT>;

template<typename Key, typename T, class Hash = hshm::hash<Key>>
using unordered_map = HSHM_NS::unordered_map<Key, T, Hash, AllocT>;

template<typename T>
using vector = HSHM_NS::vector<T, AllocT>;

template<typename T>
using key_set = HSHM_NS::KeySet<T, AllocT>;

template<typename T>
using key_queue = HSHM_NS::KeyQueue<T, AllocT>;

}