//#include "hermes_shm/data_structures/all.h"
//
//typedef HSHM_NS::Allocator AllocT;

namespace NS {
template <int LENGTH, bool WithNull>
using chararr_templ = HSHM_NS::chararr_templ<LENGTH, WithNull>;

using HSHM_NS::chararr;

template<typename T, typename AllocT = ALLOC_T>
using iqueue = HSHM_NS::iqueue<T, AllocT>;

template <typename T, typename AllocT = ALLOC_T>
using list = HSHM_NS::list<T, AllocT>;

template <typename FirstT, typename SecondT, typename AllocT = ALLOC_T>
using pair = HSHM_NS::pair<FirstT, SecondT, AllocT>;

template <typename T, typename AllocT = ALLOC_T>
using spsc_queue = HSHM_NS::spsc_queue<T, AllocT>;
template <typename T, typename AllocT = ALLOC_T>
using mpsc_queue = HSHM_NS::mpsc_queue<T, AllocT>;
template <typename T, typename AllocT = ALLOC_T>
using fixed_spsc_queue = HSHM_NS::fixed_spsc_queue<T, AllocT>;
template <typename T, typename AllocT = ALLOC_T>
using fixed_mpsc_queue = HSHM_NS::fixed_mpsc_queue<T, AllocT>;
template <typename T, typename AllocT = ALLOC_T>
using fixed_mpmc_queue = HSHM_NS::fixed_mpmc_queue<T, AllocT>;

template <typename T, typename AllocT = ALLOC_T>
using spsc_ptr_queue = HSHM_NS::spsc_ptr_queue<T, AllocT>;
template <typename T, typename AllocT = ALLOC_T>
using mpsc_ptr_queue = HSHM_NS::mpsc_ptr_queue<T, AllocT>;
template <typename T, typename AllocT = ALLOC_T>
using fixed_spsc_ptr_queue = HSHM_NS::fixed_spsc_ptr_queue<T, AllocT>;
template <typename T, typename AllocT = ALLOC_T>
using fixed_mpsc_ptr_queue = HSHM_NS::fixed_mpsc_ptr_queue<T, AllocT>;
template <typename T, typename AllocT = ALLOC_T>
using fixed_mpmc_ptr_queue = HSHM_NS::fixed_mpmc_ptr_queue<T, AllocT>;

template <typename T, typename AllocT = ALLOC_T>
using slist = HSHM_NS::slist<T, AllocT>;

template <typename T, typename AllocT = ALLOC_T>
using split_ticket_queue = HSHM_NS::split_ticket_queue<T, AllocT>;

template <typename AllocT = ALLOC_T>
using string = HSHM_NS::string_templ<HSHM_STRING_SSO, 0, AllocT>;

template <typename AllocT = ALLOC_T>
using charbuf = HSHM_NS::string_templ<HSHM_STRING_SSO, 0, AllocT>;

template <typename AllocT = ALLOC_T>
using charwrap =
    HSHM_NS::string_templ<HSHM_STRING_SSO, hipc::StringFlags::kWrap, AllocT>;

template <typename T, typename AllocT = ALLOC_T>
using ticket_queue = HSHM_NS::ticket_queue<T, AllocT>;

template <typename Key, typename T, class Hash = hshm::hash<Key>,
          typename AllocT = ALLOC_T>
using unordered_map = HSHM_NS::unordered_map<Key, T, Hash, AllocT>;

template <typename T, typename AllocT = ALLOC_T>
using vector = HSHM_NS::vector<T, AllocT>;

template <typename T, typename AllocT = ALLOC_T>
using key_set = HSHM_NS::key_set<T, AllocT>;

template <typename T, typename AllocT = ALLOC_T>
using key_queue = HSHM_NS::key_queue<T, AllocT>;
}