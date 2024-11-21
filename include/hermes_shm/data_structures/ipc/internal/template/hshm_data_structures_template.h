//#include "hermes_shm/data_structures/data_structure.h"
//
//typedef hipc::Allocator AllocT;

namespace NS {
template<typename T>
using iqueue = hipc::iqueue<T, AllocT>;

template<typename T>
using list = hipc::list<T, AllocT>;

template<typename FirstT, typename SecondT>
using pair = hipc::pair<FirstT, SecondT, AllocT>;

template<typename T>
using spsc_queue = hipc::spsc_queue<T, AllocT>;
template<typename T>
using mpsc_queue = hipc::mpsc_queue<T, AllocT>;
template<typename T>
using fixed_spsc_queue = hipc::fixed_spsc_queue<T, AllocT>;
template<typename T>
using fixed_mpsc_queue = hipc::fixed_mpsc_queue<T, AllocT>;
template<typename T>
using fixed_mpmc_queue = hipc::fixed_mpmc_queue<T, AllocT>;

template<typename T>
using spsc_ptr_queue = hipc::spsc_ptr_queue<T, AllocT>;
template<typename T>
using mpsc_ptr_queue = hipc::mpsc_ptr_queue<T, AllocT>;
template<typename T>
using fixed_spsc_ptr_queue = hipc::fixed_spsc_ptr_queue<T, AllocT>;
template<typename T>
using fixed_mpsc_ptr_queue = hipc::fixed_mpsc_ptr_queue<T, AllocT>;
template<typename T>
using fixed_mpmc_ptr_queue = hipc::fixed_mpmc_ptr_queue<T, AllocT>;

template<typename T>
using slist = hipc::slist<T, AllocT>;

template<typename T>
using split_ticket_queue = hipc::split_ticket_queue<T, AllocT>;

using string = hipc::string_templ<32, AllocT>;
using charbuf = hipc::string_templ<32, AllocT>;

template<typename T>
using ticket_queue = hipc::ticket_queue<T, AllocT>;

template<typename Key, typename T, class Hash = hshm::hash<Key>>
using unordered_map = hipc::unordered_map<Key, T, Hash, AllocT>;

template<typename T>
using vector = hipc::vector<T, AllocT>;

}