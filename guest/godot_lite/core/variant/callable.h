// godot-lite: Callable over std::function. Two kinds:
//   * callable_mp(instance, &Class::method): a typed member-function call,
//     arguments converted from Variant, the result converted back;
//   * Callable(object, "method"): dispatched through Object::callp, which
//     answers CALL_ERROR_INVALID_METHOD unless a class overrides it (there is
//     no ClassDB to look the name up in).
#pragma once

#include "core/variant/variant.h"

#include <functional>
#include <memory>
#include <type_traits>
#include <utility>

namespace gdl {

class Object;

class Callable {
public:
	struct CallError {
		enum Error {
			CALL_OK,
			CALL_ERROR_INVALID_METHOD,
			CALL_ERROR_INVALID_ARGUMENT,
			CALL_ERROR_TOO_MANY_ARGUMENTS,
			CALL_ERROR_TOO_FEW_ARGUMENTS,
			CALL_ERROR_INSTANCE_IS_NULL,
			CALL_ERROR_METHOD_NOT_CONST,
		};
		Error error = Error::CALL_OK;
		int argument = 0;
		int expected = 0;
	};

	using Function = std::function<void(const Variant **p_args, int p_argcount, Variant &r_ret, CallError &r_error)>;

private:
	struct Impl {
		Object *object = nullptr;
		StringName method;
		Function function;
	};
	std::shared_ptr<const Impl> _impl;

public:
	Callable() = default;
	Callable(const Object *p_object, const StringName &p_method);
	// gdl extension: wrap any function; p_object is reported by get_object().
	explicit Callable(Function p_function, const Object *p_object = nullptr, const StringName &p_method = StringName());

	void callp(const Variant **p_arguments, int p_argcount, Variant &r_return_value, CallError &r_call_error) const;

	template <typename... VarArgs>
	Variant call(VarArgs... p_args) const {
		Variant args[sizeof...(p_args) + 1] = { Variant(p_args)..., Variant() };
		const Variant *argptrs[sizeof...(p_args) + 1];
		for (size_t i = 0; i < sizeof...(p_args); i++) {
			argptrs[i] = &args[i];
		}
		Variant ret;
		CallError ce;
		callp(sizeof...(p_args) == 0 ? nullptr : argptrs, int(sizeof...(p_args)), ret, ce);
		return ret;
	}

	bool is_null() const { return !_impl; }
	bool is_valid() const;
	Object *get_object() const { return _impl ? _impl->object : nullptr; }
	StringName get_method() const { return _impl ? _impl->method : StringName(); }

	bool operator==(const Callable &p_callable) const;
	bool operator!=(const Callable &p_callable) const { return !(*this == p_callable); }
	uint32_t hash() const;
};

namespace gdl_callable_internal {

template <typename P>
std::decay_t<P> cast_arg(const Variant &p_arg) {
	using D = std::decay_t<P>;
	if constexpr (std::is_same_v<D, Variant>) {
		return p_arg;
	} else {
		return static_cast<D>(p_arg);
	}
}

template <typename R, typename F, size_t... Is>
void invoke(F &&p_f, const Variant **p_args, Variant &r_ret, std::index_sequence<Is...>) {
	if constexpr (std::is_void_v<R>) {
		p_f(p_args[Is]...);
		r_ret = Variant();
	} else {
		r_ret = Variant(p_f(p_args[Is]...));
	}
}

template <typename R, typename... P, typename Call>
Callable make(const Object *p_object, Call p_call) {
	return Callable(
			[p_call](const Variant **p_args, int p_argcount, Variant &r_ret, Callable::CallError &r_error) {
				constexpr int N = int(sizeof...(P));
				if (p_argcount < N) {
					r_error.error = Callable::CallError::CALL_ERROR_TOO_FEW_ARGUMENTS;
					r_error.expected = N;
					return;
				}
				if (p_argcount > N) {
					r_error.error = Callable::CallError::CALL_ERROR_TOO_MANY_ARGUMENTS;
					r_error.expected = N;
					return;
				}
				r_error.error = Callable::CallError::CALL_OK;
				invoke<R>(p_call, p_args, r_ret, std::index_sequence_for<P...>{});
			},
			p_object);
}

} // namespace gdl_callable_internal

template <typename T, typename C, typename R, typename... P>
Callable callable_mp(T *p_instance, R (C::*p_method)(P...)) {
	C *inst = static_cast<C *>(p_instance);
	return gdl_callable_internal::make<R, P...>(inst, [inst, p_method](auto... p_args) -> R {
		return (inst->*p_method)(gdl_callable_internal::cast_arg<P>(*p_args)...);
	});
}

template <typename T, typename C, typename R, typename... P>
Callable callable_mp(T *p_instance, R (C::*p_method)(P...) const) {
	const C *inst = static_cast<const C *>(p_instance);
	return gdl_callable_internal::make<R, P...>(inst, [inst, p_method](auto... p_args) -> R {
		return (inst->*p_method)(gdl_callable_internal::cast_arg<P>(*p_args)...);
	});
}

} // namespace gdl
