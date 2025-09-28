namespace NS {
template <int LENGTH, bool WithNull>
using chararr_templ = HSHM_NS::chararr_templ<LENGTH, WithNull>;

using HSHM_NS::chararr;

template <typename T>
using lifo_list_queue = HSHM_NS::lifo_list_queue<T, ALLOC_T>;

template <typename T>
using list = HSHM_NS::list<T, ALLOC_T>;

template <typename T>
using mpsc_lifo_list_queue = HSHM_NS::mpsc_lifo_list_queue<T, ALLOC_T>;

template <typename T>
using spsc_fifo_list_queue = HSHM_NS::spsc_fifo_list_queue<T, ALLOC_T>;

template <typename FirstT, typename SecondT>
using pair = HSHM_NS::pair<FirstT, SecondT, ALLOC_T>;

template <typename T, typename HDR = HSHM_NS::EmptyHeader>
using spsc_queue = HSHM_NS::spsc_queue<T, HDR, ALLOC_T>;
template <typename T, typename HDR = HSHM_NS::EmptyHeader>
using mpsc_queue = HSHM_NS::mpsc_queue<T, HDR, ALLOC_T>;
template <typename T, typename HDR = HSHM_NS::EmptyHeader>
using fixed_spsc_queue = HSHM_NS::fixed_spsc_queue<T, HDR, ALLOC_T>;
template <typename T, typename HDR = HSHM_NS::EmptyHeader>
using fixed_mpsc_queue = HSHM_NS::fixed_mpsc_queue<T, HDR, ALLOC_T>;
template <typename T, typename HDR = HSHM_NS::EmptyHeader>
using circular_mpsc_queue = HSHM_NS::circular_mpsc_queue<T, HDR, ALLOC_T>;
template <typename T, typename HDR = HSHM_NS::EmptyHeader>
using circular_spsc_queue = HSHM_NS::circular_spsc_queue<T, HDR, ALLOC_T>;
template <typename T, typename HDR = HSHM_NS::EmptyHeader>
using ext_ring_buffer = HSHM_NS::ext_ring_buffer<T, HDR, ALLOC_T>;

template <typename T, typename HDR = HSHM_NS::EmptyHeader>
using spsc_ptr_queue = HSHM_NS::spsc_ptr_queue<T, HDR, ALLOC_T>;
template <typename T, typename HDR = HSHM_NS::EmptyHeader>
using mpsc_ptr_queue = HSHM_NS::mpsc_ptr_queue<T, HDR, ALLOC_T>;
template <typename T, typename HDR = HSHM_NS::EmptyHeader>
using fixed_spsc_ptr_queue = HSHM_NS::fixed_spsc_ptr_queue<T, HDR, ALLOC_T>;
template <typename T, typename HDR = HSHM_NS::EmptyHeader>
using fixed_mpsc_ptr_queue = HSHM_NS::fixed_mpsc_ptr_queue<T, HDR, ALLOC_T>;
template <typename T, typename HDR = HSHM_NS::EmptyHeader>
using circular_mpsc_ptr_queue = HSHM_NS::circular_mpsc_ptr_queue<T, HDR, ALLOC_T>;
template <typename T, typename HDR = HSHM_NS::EmptyHeader>
using circular_spsc_ptr_queue = HSHM_NS::circular_spsc_ptr_queue<T, HDR, ALLOC_T>;
template <typename T, typename HDR = HSHM_NS::EmptyHeader>
using ext_ptr_ring_buffer = HSHM_NS::ext_ptr_ring_buffer<T, HDR, ALLOC_T>;

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
using spsc_key_set = HSHM_NS::spsc_key_set<T, ALLOC_T>;

template <typename T>
using mpmc_key_set = HSHM_NS::mpmc_key_set<T, ALLOC_T>;

template <typename T>
using dynamic_queue = HSHM_NS::dynamic_queue<T, ALLOC_T>;
}  // namespace NS