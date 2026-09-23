// tbb_serial: a header-only, single-threaded stand-in for the part of oneTBB
// that fit.elf's dependencies call (ipc-toolkit, scalable-ccd; PolyFEM itself
// builds with POLYFEM_THREADING=NONE and calls none of it).
//
// Why it exists: ipc-toolkit and scalable-ccd link TBB::tbb unconditionally,
// and oneTBB's runtime (a thread pool, a scheduler, a malloc) has nothing to
// run on in the guest -- threads there are serialised (Gate 0C) and the heap
// is the sandbox's. cmake/tbb_serial.cmake defines TBB::tbb as this include
// directory, so the recipes' `if(TARGET TBB::tbb) return()` guards skip oneTBB
// entirely and nothing of oneTBB is compiled or linked.
//
// Semantics are those of oneTBB held to one thread
// (global_control(max_allowed_parallelism, 1)), which is how the native
// oracle runs (tools/fit/fit_native.cpp):
//   - parallel_for(range, body) calls body once on the whole range, or not at
//     all when the range is empty. oneTBB on one thread calls it on sub-ranges
//     left to right; the bodies in this build accumulate element by element
//     into one thread-local, so the two orders give the same bits (checked:
//     fit_native with FIT_TBB=serial reproduces its oneTBB f64 run bitwise).
//   - parallel_for(first, last[, step], f) calls f(i) in index order.
//   - parallel_reduce(range, identity, body, join) is body(range, identity).
//   - enumerable_thread_specific / combinable hold at most one element, made
//     on the first local(); combine() of none returns a fresh element, of one
//     returns it (oneTBB's rule).
//   - parallel_sort is std::sort. oneTBB's parallel_sort is not stable
//     either; with a comparator that has ties (scalable-ccd's SortBoxes, used
//     only by the sweep-and-prune broad phase) the order of equal elements
//     may differ from oneTBB's. The fit uses the BVH broad phase, whose
//     candidates are sorted with a total order.
//   - global_control, task_arena, info report one thread.
// Anything else from oneTBB fails to compile, which is the point: a new use
// has to be looked at before it is added here.
#pragma once

#include <algorithm>
#include <cstddef>
#include <functional>
#include <iterator>
#include <memory>
#include <tuple>
#include <type_traits>
#include <utility>

#define TBB_SERIAL_STANDIN 1

namespace tbb {

// --- ranges -----------------------------------------------------------------

class split {};

template <typename Value>
class blocked_range {
public:
	using const_iterator = Value;
	using size_type = std::size_t;

	blocked_range() = default;
	blocked_range(Value begin, Value end, size_type grainsize = 1) :
			b_(begin), e_(end), g_(grainsize) {}
	// Splitting constructor, for API completeness; the stand-in never splits.
	blocked_range(blocked_range &r, split) :
			b_(r.b_ + (r.e_ - r.b_) / 2), e_(r.e_), g_(r.g_) { r.e_ = b_; }

	const_iterator begin() const { return b_; }
	const_iterator end() const { return e_; }
	size_type size() const { return size_type(e_ - b_); }
	size_type grainsize() const { return g_; }
	bool empty() const { return !(b_ < e_); }
	bool is_divisible() const { return false; }

private:
	Value b_{}, e_{};
	size_type g_ = 1;
};

template <typename RowValue, typename ColValue = RowValue>
class blocked_range2d {
public:
	using row_range_type = blocked_range<RowValue>;
	using col_range_type = blocked_range<ColValue>;

	blocked_range2d(RowValue row_begin, RowValue row_end, std::size_t row_grain,
			ColValue col_begin, ColValue col_end, std::size_t col_grain) :
			rows_(row_begin, row_end, row_grain), cols_(col_begin, col_end, col_grain) {}
	blocked_range2d(RowValue row_begin, RowValue row_end, ColValue col_begin, ColValue col_end) :
			rows_(row_begin, row_end), cols_(col_begin, col_end) {}

	bool empty() const { return rows_.empty() || cols_.empty(); }
	bool is_divisible() const { return false; }
	const row_range_type &rows() const { return rows_; }
	const col_range_type &cols() const { return cols_; }

private:
	row_range_type rows_;
	col_range_type cols_;
};

// --- partitioners (accepted and ignored) -------------------------------------

class auto_partitioner {};
class simple_partitioner {};
class static_partitioner {};
class affinity_partitioner {};

namespace serial_detail {
template <typename P>
struct is_partitioner : std::false_type {};
template <> struct is_partitioner<auto_partitioner> : std::true_type {};
template <> struct is_partitioner<simple_partitioner> : std::true_type {};
template <> struct is_partitioner<static_partitioner> : std::true_type {};
template <> struct is_partitioner<affinity_partitioner> : std::true_type {};
template <typename P>
constexpr bool is_partitioner_v = is_partitioner<std::remove_cv_t<std::remove_reference_t<P>>>::value;
} // namespace serial_detail

// --- parallel_for ------------------------------------------------------------

template <typename Range, typename Body>
void parallel_for(const Range &range, const Body &body) {
	if (!range.empty())
		body(range);
}

template <typename Range, typename Body, typename Partitioner,
		std::enable_if_t<serial_detail::is_partitioner_v<Partitioner>, int> = 0>
void parallel_for(const Range &range, const Body &body, Partitioner &&) {
	if (!range.empty())
		body(range);
}

template <typename Index, typename Function,
		std::enable_if_t<std::is_integral_v<Index> && std::is_invocable_v<const Function &, Index>, int> = 0>
void parallel_for(Index first, Index last, const Function &f) {
	for (Index i = first; i < last; ++i)
		f(i);
}

template <typename Index, typename Function,
		std::enable_if_t<std::is_integral_v<Index> && std::is_invocable_v<const Function &, Index>, int> = 0>
void parallel_for(Index first, Index last, Index step, const Function &f) {
	for (Index i = first; i < last; i += step)
		f(i);
}

// --- parallel_reduce ---------------------------------------------------------

template <typename Range, typename Value, typename RealBody, typename Reduction>
Value parallel_reduce(const Range &range, const Value &identity, const RealBody &real_body, const Reduction &) {
	if (range.empty())
		return identity;
	return real_body(range, identity);
}

template <typename Range, typename Value, typename RealBody, typename Reduction, typename Partitioner,
		std::enable_if_t<serial_detail::is_partitioner_v<Partitioner>, int> = 0>
Value parallel_reduce(const Range &range, const Value &identity, const RealBody &real_body, const Reduction &,
		Partitioner &&) {
	if (range.empty())
		return identity;
	return real_body(range, identity);
}

// Imperative form: Body has operator()(const Range&) and join(Body&).
template <typename Range, typename Body>
void parallel_reduce(const Range &range, Body &body) {
	if (!range.empty())
		body(range);
}

// --- parallel_sort -----------------------------------------------------------

template <typename RandomIt>
void parallel_sort(RandomIt first, RandomIt last) {
	std::sort(first, last);
}

template <typename RandomIt, typename Compare>
void parallel_sort(RandomIt first, RandomIt last, const Compare &comp) {
	std::sort(first, last, comp);
}

template <typename Container>
void parallel_sort(Container &c) {
	std::sort(std::begin(c), std::end(c));
}

// --- thread-local storage ----------------------------------------------------

namespace serial_detail {
// One lazily made element. make_ builds it the way oneTBB's construct
// callback would: default, from an exemplar copy, from a finit functor, or
// from stored constructor arguments.
template <typename T>
class one_slot {
public:
	using iterator = T *;
	using const_iterator = const T *;

	one_slot() : make_([] { return std::make_unique<T>(); }) {}
	explicit one_slot(std::function<std::unique_ptr<T>()> make) : make_(std::move(make)) {}
	one_slot(const one_slot &o) : make_(o.make_), slot_(o.slot_ ? std::make_unique<T>(*o.slot_) : nullptr) {}
	one_slot &operator=(const one_slot &o) {
		if (this != &o) {
			make_ = o.make_;
			slot_ = o.slot_ ? std::make_unique<T>(*o.slot_) : nullptr;
		}
		return *this;
	}
	one_slot(one_slot &&) = default;
	one_slot &operator=(one_slot &&) = default;

	T &local() {
		if (!slot_)
			slot_ = make_();
		return *slot_;
	}
	T &local(bool &exists) {
		exists = bool(slot_);
		return local();
	}
	std::size_t size() const { return slot_ ? 1 : 0; }
	bool empty() const { return !slot_; }
	void clear() { slot_.reset(); }

	iterator begin() { return slot_.get(); }
	iterator end() { return slot_.get() + size(); }
	const_iterator begin() const { return slot_.get(); }
	const_iterator end() const { return slot_.get() + size(); }

	template <typename F>
	T combine(F &&) const {
		if (!slot_)
			return *make_();
		return *slot_;
	}
	template <typename F>
	void combine_each(F &&f) const {
		if (slot_)
			f(*slot_);
	}
	template <typename F>
	void combine_each(F &&f) {
		if (slot_)
			f(*slot_);
	}

private:
	std::function<std::unique_ptr<T>()> make_;
	std::unique_ptr<T> slot_;
};
} // namespace serial_detail

enum ets_key_usage_type { ets_key_per_instance, ets_no_key, ets_suspend_aware };

template <typename T, typename Allocator = void, ets_key_usage_type ETS_key_type = ets_no_key>
class enumerable_thread_specific : public serial_detail::one_slot<T> {
	using base = serial_detail::one_slot<T>;

	template <typename U>
	static constexpr bool is_self_v = std::is_same_v<std::decay_t<U>, enumerable_thread_specific>;

public:
	using value_type = T;
	using reference = T &;
	using const_reference = const T &;
	using pointer = T *;
	using size_type = std::size_t;
	using difference_type = std::ptrdiff_t;
	using iterator = typename base::iterator;
	using const_iterator = typename base::const_iterator;

	enumerable_thread_specific() = default;

	// Exemplar: every element starts as a copy of it.
	explicit enumerable_thread_specific(const T &exemplar) :
			base([exemplar] { return std::make_unique<T>(exemplar); }) {}
	explicit enumerable_thread_specific(T &&exemplar) :
			base([ex = std::move(exemplar)] { return std::make_unique<T>(ex); }) {}

	// Finit: a functor returning T.
	template <typename Finit,
			std::enable_if_t<!is_self_v<Finit> && !std::is_same_v<std::decay_t<Finit>, T> &&
							std::is_invocable_r_v<T, Finit &>,
					int> = 0>
	explicit enumerable_thread_specific(Finit finit) :
			base([finit]() mutable { return std::make_unique<T>(finit()); }) {}

	// Constructor arguments: every element is T(args...).
	template <typename P1, typename... P,
			std::enable_if_t<!is_self_v<P1> && !std::is_same_v<std::decay_t<P1>, T> &&
							!(sizeof...(P) == 0 && std::is_invocable_r_v<T, std::decay_t<P1> &>),
					int> = 0>
	enumerable_thread_specific(P1 &&arg1, P &&...args) :
			base([tup = std::make_tuple(std::forward<P1>(arg1), std::forward<P>(args)...)] {
				return std::apply([](const auto &...a) { return std::make_unique<T>(a...); }, tup);
			}) {}

	enumerable_thread_specific(const enumerable_thread_specific &) = default;
	enumerable_thread_specific(enumerable_thread_specific &&) = default;
	enumerable_thread_specific &operator=(const enumerable_thread_specific &) = default;
	enumerable_thread_specific &operator=(enumerable_thread_specific &&) = default;
};

template <typename T>
class combinable : public serial_detail::one_slot<T> {
	using base = serial_detail::one_slot<T>;

public:
	combinable() = default;
	template <typename Finit,
			std::enable_if_t<!std::is_same_v<std::decay_t<Finit>, combinable> && std::is_invocable_r_v<T, Finit &>, int> = 0>
	explicit combinable(Finit finit) :
			base([finit]() mutable { return std::make_unique<T>(finit()); }) {}
};

// --- scheduler controls: one thread --------------------------------------------

class global_control {
public:
	enum parameter {
		max_allowed_parallelism,
		thread_stack_size,
		terminate_on_exception,
		scheduler_handle,
		parameter_max
	};
	global_control(parameter, std::size_t) {}
	static std::size_t active_value(parameter p) { return p == max_allowed_parallelism ? 1 : 0; }
};

namespace info {
inline int default_concurrency() { return 1; }
} // namespace info

class task_arena {
public:
	static constexpr int automatic = -1;
	explicit task_arena(int = automatic, unsigned = 1) {}
	void initialize() {}
	template <typename F>
	auto execute(F &&f) -> decltype(f()) { return f(); }
	int max_concurrency() const { return 1; }
	static int current_thread_index() { return 0; }
};

namespace this_task_arena {
inline int current_thread_index() { return 0; }
inline int max_concurrency() { return 1; }
} // namespace this_task_arena

} // namespace tbb
