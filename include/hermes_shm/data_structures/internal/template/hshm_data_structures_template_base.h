// #include "hermes_shm/data_structures/all.h"
//
// typedef HSHM_NS::Allocator ALLOC_T;

namespace NS {
template <int LENGTH, bool WithNull>
using chararr_templ = HSHM_NS::chararr_templ<LENGTH, WithNull>;

using HSHM_NS::chararr;

template <typename T>
using iqueue = HSHM_NS::iqueue<T, ALLOC_T>;

template <typename T>
using list = HSHM_NS::list<T, ALLOC_T>;

template <typename FirstT, typename SecondT>
using pair = HSHM_NS::pair<FirstT, SecondT, ALLOC_T>;

template <typename T>
using spsc_queue = HSHM_NS::spsc_queue<T, ALLOC_T>;
template <typename T>
using mpsc_queue = HSHM_NS::mpsc_queue<T, ALLOC_T>;
template <typename T>
using fixed_spsc_queue = HSHM_NS::fixed_spsc_queue<T, ALLOC_T>;
template <typename T>
using fixed_mpsc_queue = HSHM_NS::fixed_mpsc_queue<T, ALLOC_T>;
template <typename T>
using fixed_mpmc_queue = HSHM_NS::fixed_mpmc_queue<T, ALLOC_T>;

template <typename T>
using spsc_ptr_queue = HSHM_NS::spsc_ptr_queue<T, ALLOC_T>;
template <typename T>
using mpsc_ptr_queue = HSHM_NS::mpsc_ptr_queue<T, ALLOC_T>;
template <typename T>
using fixed_spsc_ptr_queue = HSHM_NS::fixed_spsc_ptr_queue<T, ALLOC_T>;
template <typename T>
using fixed_mpsc_ptr_queue = HSHM_NS::fixed_mpsc_ptr_queue<T, ALLOC_T>;
template <typename T>
using fixed_mpmc_ptr_queue = HSHM_NS::fixed_mpmc_ptr_queue<T, ALLOC_T>;

template <typename T>
using slist = HSHM_NS::slist<T, ALLOC_T>;

template <typename T>
using split_ticket_queue = HSHM_NS::split_ticket_queue<T, ALLOC_T>;

using string = HSHM_NS::string_templ<HSHM_STRING_SSO, 0, ALLOC_T>;

using charbuf = HSHM_NS::string_templ<HSHM_STRING_SSO, 0, ALLOC_T>;

using charwrap =
    HSHM_NS::string_templ<HSHM_STRING_SSO, hipc::StringFlags::kWrap, ALLOC_T>;

template <typename T>
using ticket_queue = HSHM_NS::ticket_queue<T, ALLOC_T>;

template <typename Key, typename T, class Hash = hshm::hash<Key>>
using unordered_map = HSHM_NS::unordered_map<Key, T, Hash, ALLOC_T>;

template <typename T>
using vector = HSHM_NS::vector<T, ALLOC_T>;

template <typename T>
using key_set = HSHM_NS::key_set<T, ALLOC_T>;

template <typename T>
using key_queue = HSHM_NS::key_queue<T, ALLOC_T>;
}  // namespace NS