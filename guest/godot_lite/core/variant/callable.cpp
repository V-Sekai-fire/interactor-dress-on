#include "core/variant/callable.h"

#include "core/object/object.h"
#include "core/templates/hashfuncs.h"

namespace gdl {

Callable::Callable(const Object *p_object, const StringName &p_method) {
	auto impl = std::make_shared<Impl>();
	impl->object = const_cast<Object *>(p_object);
	impl->method = p_method;
	_impl = impl;
}

Callable::Callable(Function p_function, const Object *p_object, const StringName &p_method) {
	auto impl = std::make_shared<Impl>();
	impl->object = const_cast<Object *>(p_object);
	impl->method = p_method;
	impl->function = std::move(p_function);
	_impl = impl;
}

void Callable::callp(const Variant **p_arguments, int p_argcount, Variant &r_return_value, CallError &r_call_error) const {
	r_call_error = CallError();
	r_return_value = Variant();
	if (!_impl) {
		r_call_error.error = CallError::CALL_ERROR_INSTANCE_IS_NULL;
		return;
	}
	if (_impl->function) {
		_impl->function(p_arguments, p_argcount, r_return_value, r_call_error);
		return;
	}
	if (!_impl->object) {
		r_call_error.error = CallError::CALL_ERROR_INSTANCE_IS_NULL;
		return;
	}
	r_return_value = _impl->object->callp(_impl->method, p_arguments, p_argcount, r_call_error);
}

bool Callable::is_valid() const {
	return _impl && (_impl->function || _impl->object != nullptr);
}

bool Callable::operator==(const Callable &p_callable) const {
	if (_impl == p_callable._impl) {
		return true;
	}
	if (!_impl || !p_callable._impl) {
		return false;
	}
	// Two standard callables naming the same method on the same object.
	return !_impl->function && !p_callable._impl->function && _impl->object == p_callable._impl->object && _impl->method == p_callable._impl->method;
}

uint32_t Callable::hash() const {
	if (!_impl) {
		return 0;
	}
	if (_impl->function) {
		return hash_one_uint64(uint64_t(uintptr_t(_impl.get())));
	}
	return hash_murmur3_one_32(_impl->method.hash(), hash_one_uint64(uint64_t(uintptr_t(_impl->object))));
}

} // namespace gdl
