// godot-lite: the ERR_* / WARN_* / CRASH_* macros that Cassie and the
// vendored core use, with Godot's control flow (return / abort) and messages
// printed to stderr. No editor notification, no error handler list. A macro
// missing here is one nothing uses yet.
#pragma once

#include "core/typedefs.h"

#include <cstdint>

namespace gdl {

class String;

enum ErrorHandlerType {
	ERR_HANDLER_ERROR,
	ERR_HANDLER_WARNING,
};

void _err_print_error(const char *p_function, const char *p_file, int p_line, const char *p_error, const char *p_message = "", ErrorHandlerType p_type = ERR_HANDLER_ERROR);
void _err_print_error(const char *p_function, const char *p_file, int p_line, const char *p_error, const String &p_message, ErrorHandlerType p_type = ERR_HANDLER_ERROR);
void _err_print_error(const char *p_function, const char *p_file, int p_line, const String &p_error, const char *p_message = "", ErrorHandlerType p_type = ERR_HANDLER_ERROR);
void _err_print_index_error(const char *p_function, const char *p_file, int p_line, int64_t p_index, int64_t p_size, const char *p_index_str, const char *p_size_str, const char *p_message = "", bool p_fatal = false);
void _err_print_index_error(const char *p_function, const char *p_file, int p_line, int64_t p_index, int64_t p_size, const char *p_index_str, const char *p_size_str, const String &p_message, bool p_fatal = false);
[[noreturn]] void _err_crash();

} // namespace gdl

#define FUNCTION_STR __FUNCTION__

#define _GDL_ERR(m_err, m_msg) ::gdl::_err_print_error(FUNCTION_STR, __FILE__, __LINE__, m_err, m_msg)
#define _GDL_IDX(m_index, m_size, m_msg, m_fatal) ::gdl::_err_print_index_error(FUNCTION_STR, __FILE__, __LINE__, int64_t(m_index), int64_t(m_size), _STR(m_index), _STR(m_size), m_msg, m_fatal)

// Index checks.
#define ERR_FAIL_INDEX(m_index, m_size)                     \
	if (unlikely((m_index) < 0 || (m_index) >= (m_size))) { \
		_GDL_IDX(m_index, m_size, "", false);               \
		return;                                             \
	} else                                                  \
		((void)0)
#define ERR_FAIL_INDEX_V(m_index, m_size, m_retval) ERR_FAIL_INDEX_V_MSG(m_index, m_size, m_retval, "")
#define ERR_FAIL_INDEX_V_MSG(m_index, m_size, m_retval, m_msg) \
	if (unlikely((m_index) < 0 || (m_index) >= (m_size))) {    \
		_GDL_IDX(m_index, m_size, m_msg, false);               \
		return m_retval;                                       \
	} else                                                     \
		((void)0)
#define CRASH_BAD_INDEX(m_index, m_size)                    \
	if (unlikely((m_index) < 0 || (m_index) >= (m_size))) { \
		_GDL_IDX(m_index, m_size, "", true);                \
		::gdl::_err_crash();                                \
	} else                                                  \
		((void)0)

#define ERR_FAIL_UNSIGNED_INDEX(m_index, m_size) \
	if (unlikely((m_index) >= (m_size))) {       \
		_GDL_IDX(m_index, m_size, "", false);    \
		return;                                  \
	} else                                       \
		((void)0)
#define ERR_FAIL_UNSIGNED_INDEX_V(m_index, m_size, m_retval) \
	if (unlikely((m_index) >= (m_size))) {                   \
		_GDL_IDX(m_index, m_size, "", false);                \
		return m_retval;                                     \
	} else                                                   \
		((void)0)
#define CRASH_BAD_UNSIGNED_INDEX(m_index, m_size) \
	if (unlikely((m_index) >= (m_size))) {        \
		_GDL_IDX(m_index, m_size, "", true);      \
		::gdl::_err_crash();                      \
	} else                                        \
		((void)0)

// Null checks.
#define ERR_FAIL_NULL_V(m_param, m_retval) ERR_FAIL_NULL_V_MSG(m_param, m_retval, "")
#define ERR_FAIL_NULL_V_MSG(m_param, m_retval, m_msg)                   \
	if (unlikely(m_param == nullptr)) {                                 \
		_GDL_ERR("Parameter \"" _STR(m_param) "\" is null.", m_msg); \
		return m_retval;                                                \
	} else                                                              \
		((void)0)

// Condition checks.
#define ERR_FAIL_COND(m_cond) ERR_FAIL_COND_MSG(m_cond, "")
#define ERR_FAIL_COND_MSG(m_cond, m_msg)                               \
	if (unlikely(m_cond)) {                                            \
		_GDL_ERR("Condition \"" _STR(m_cond) "\" is true.", m_msg); \
		return;                                                        \
	} else                                                             \
		((void)0)
#define ERR_FAIL_COND_V(m_cond, m_retval) ERR_FAIL_COND_V_MSG(m_cond, m_retval, "")
#define ERR_FAIL_COND_V_MSG(m_cond, m_retval, m_msg)                                                  \
	if (unlikely(m_cond)) {                                                                           \
		_GDL_ERR("Condition \"" _STR(m_cond) "\" is true. Returning: " _STR(m_retval), m_msg); \
		return m_retval;                                                                              \
	} else                                                                                            \
		((void)0)
#define CRASH_COND(m_cond) CRASH_COND_MSG(m_cond, "")
#define CRASH_COND_MSG(m_cond, m_msg)                                         \
	if (unlikely(m_cond)) {                                                   \
		_GDL_ERR("FATAL: Condition \"" _STR(m_cond) "\" is true.", m_msg); \
		::gdl::_err_crash();                                                  \
	} else                                                                    \
		((void)0)

// Unconditional.
#define ERR_FAIL_MSG(m_msg)                          \
	if (true) {                                      \
		_GDL_ERR("Method/function failed.", m_msg); \
		return;                                      \
	} else                                           \
		((void)0)
#define ERR_FAIL_V(m_retval) ERR_FAIL_V_MSG(m_retval, "")
#define ERR_FAIL_V_MSG(m_retval, m_msg)                                          \
	if (true) {                                                                  \
		_GDL_ERR("Method/function failed. Returning: " _STR(m_retval), m_msg); \
		return m_retval;                                                         \
	} else                                                                       \
		((void)0)
#define ERR_PRINT(m_msg) _GDL_ERR(m_msg, "")
#define WARN_PRINT(m_msg) ::gdl::_err_print_error(FUNCTION_STR, __FILE__, __LINE__, m_msg, "", ::gdl::ERR_HANDLER_WARNING)

// Godot only checks DEV_ASSERT in dev builds; neither target is one.
#define DEV_ASSERT(m_cond) ((void)0)
