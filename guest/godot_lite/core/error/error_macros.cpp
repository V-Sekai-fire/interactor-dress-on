#include "core/error/error_macros.h"

#include "core/core_globals.h"
#include "core/string/ustring.h"

#include <cstdio>
#include <cstdlib>

namespace gdl {

static void _print(const char *p_function, const char *p_file, int p_line, const char *p_error, const char *p_message, ErrorHandlerType p_type) {
	if (!CoreGlobals::print_error_enabled) {
		return;
	}
	const char *kind = p_type == ERR_HANDLER_WARNING ? "WARNING" : "ERROR";
	if (p_message && p_message[0]) {
		fprintf(stderr, "%s: %s\n   at: %s (%s:%d) - %s\n", kind, p_message, p_function, p_file, p_line, p_error);
	} else {
		fprintf(stderr, "%s: %s\n   at: %s (%s:%d)\n", kind, p_error, p_function, p_file, p_line);
	}
	fflush(stderr);
}

void _err_print_error(const char *p_function, const char *p_file, int p_line, const char *p_error, const char *p_message, ErrorHandlerType p_type) {
	_print(p_function, p_file, p_line, p_error, p_message, p_type);
}

void _err_print_error(const char *p_function, const char *p_file, int p_line, const char *p_error, const String &p_message, ErrorHandlerType p_type) {
	_print(p_function, p_file, p_line, p_error, p_message.get_data(), p_type);
}

void _err_print_error(const char *p_function, const char *p_file, int p_line, const String &p_error, const char *p_message, ErrorHandlerType p_type) {
	_print(p_function, p_file, p_line, p_error.get_data(), p_message, p_type);
}

void _err_print_index_error(const char *p_function, const char *p_file, int p_line, int64_t p_index, int64_t p_size, const char *p_index_str, const char *p_size_str, const char *p_message, bool p_fatal) {
	char err[512];
	snprintf(err, sizeof(err), "%sIndex %s = %lld is out of bounds (%s = %lld).", p_fatal ? "FATAL: " : "", p_index_str, (long long)p_index, p_size_str, (long long)p_size);
	_print(p_function, p_file, p_line, err, p_message, ERR_HANDLER_ERROR);
}

void _err_print_index_error(const char *p_function, const char *p_file, int p_line, int64_t p_index, int64_t p_size, const char *p_index_str, const char *p_size_str, const String &p_message, bool p_fatal) {
	_err_print_index_error(p_function, p_file, p_line, p_index, p_size, p_index_str, p_size_str, p_message.get_data(), p_fatal);
}

void _err_crash() {
	fflush(stdout);
	fflush(stderr);
	abort();
}

} // namespace gdl
