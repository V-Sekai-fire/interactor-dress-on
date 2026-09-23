// godot-lite: a minimal String over std::string (UTF-8 bytes, not Godot's
// char32_t). It carries what Cassie and the vendored core use: construction
// from C strings, concatenation, comparison, hashing for HashMap/HashSet,
// to_float/to_int, sprintf for vformat, and the number formatters.
// StringName is the same type: nothing here interns names.
#pragma once

#include "core/templates/hashfuncs.h"
#include "core/typedefs.h"

#include <cstdint>
#include <string>

namespace gdl {

class Array;

class CharString {
	std::string data;

public:
	CharString() = default;
	explicit CharString(std::string p_data) :
			data(std::move(p_data)) {}
	const char *get_data() const { return data.c_str(); }
	const char *ptr() const { return data.c_str(); }
	int length() const { return int(data.size()); }
	int size() const { return data.empty() ? 0 : int(data.size()) + 1; }
};

class String {
	std::string _s;

public:
	String() = default;
	String(const char *p_str) :
			_s(p_str ? p_str : "") {}
	String(const std::string &p_str) :
			_s(p_str) {}
	String(std::string &&p_str) :
			_s(std::move(p_str)) {}

	// gdl extension: the backing UTF-8 buffer.
	const std::string &std_string() const { return _s; }
	const char *get_data() const { return _s.c_str(); }

	bool is_empty() const { return _s.empty(); }
	int length() const { return int(_s.size()); }
	int size() const { return _s.empty() ? 0 : int(_s.size()) + 1; }
	void clear() { _s.clear(); }

	CharString utf8() const { return CharString(_s); }

	uint32_t hash() const;

	bool operator==(const String &p_o) const { return _s == p_o._s; }
	bool operator!=(const String &p_o) const { return _s != p_o._s; }
	bool operator<(const String &p_o) const { return _s < p_o._s; }
	bool operator<=(const String &p_o) const { return _s <= p_o._s; }
	bool operator>(const String &p_o) const { return _s > p_o._s; }
	bool operator>=(const String &p_o) const { return _s >= p_o._s; }
	bool operator==(const char *p_o) const { return _s == (p_o ? p_o : ""); }
	bool operator!=(const char *p_o) const { return !(*this == p_o); }

	String operator+(const String &p_o) const { return String(_s + p_o._s); }
	String operator+(const char *p_o) const { return String(_s + (p_o ? p_o : "")); }
	String operator+(char32_t p_c) const;
	String &operator+=(const String &p_o) {
		_s += p_o._s;
		return *this;
	}
	String &operator+=(const char *p_o) {
		_s += (p_o ? p_o : "");
		return *this;
	}
	String &operator+=(char32_t p_c);

	double to_float() const;
	int64_t to_int() const;

	// Godot's printf-alike (%d %i %s %f %e %g %x %X %c %%, flags, width, precision);
	// the arguments come from an Array of Variants.
	String sprintf(const Array &p_values, bool *r_error) const;

	static String num(double p_num, int p_decimals = -1);
	static String num_int64(int64_t p_num, int p_base = 10, bool p_capitalize_hex = false);
	static String num_real(double p_num, bool p_trailing = true);
};

using StringName = String;

String operator+(const char *p_a, const String &p_b);
String operator+(char32_t p_a, const String &p_b);
bool operator==(const char *p_a, const String &p_b);
bool operator!=(const char *p_a, const String &p_b);

String itos(int64_t p_val);
String rtos(double p_val);

} // namespace gdl
