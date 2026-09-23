#include "core/string/ustring.h"

#include "core/core_globals.h"
#include "core/string/print_string.h"
#include "core/templates/hashfuncs.h"
#include "core/variant/array.h"
#include "core/variant/variant.h"

#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace gdl {

static void _append_utf8(std::string &r_s, char32_t p_c) {
	if (p_c < 0x80) {
		r_s += char(p_c);
	} else if (p_c < 0x800) {
		r_s += char(0xC0 | (p_c >> 6));
		r_s += char(0x80 | (p_c & 0x3F));
	} else if (p_c < 0x10000) {
		r_s += char(0xE0 | (p_c >> 12));
		r_s += char(0x80 | ((p_c >> 6) & 0x3F));
		r_s += char(0x80 | (p_c & 0x3F));
	} else {
		r_s += char(0xF0 | (p_c >> 18));
		r_s += char(0x80 | ((p_c >> 12) & 0x3F));
		r_s += char(0x80 | ((p_c >> 6) & 0x3F));
		r_s += char(0x80 | (p_c & 0x3F));
	}
}

uint32_t String::hash() const {
	return hash_djb2_buffer((const uint8_t *)_s.data(), int(_s.size()));
}

String String::operator+(char32_t p_c) const {
	String r = *this;
	_append_utf8(r._s, p_c);
	return r;
}

String &String::operator+=(char32_t p_c) {
	_append_utf8(_s, p_c);
	return *this;
}

double String::to_float() const {
	return strtod(_s.c_str(), nullptr);
}

int64_t String::to_int() const {
	return strtoll(_s.c_str(), nullptr, 10);
}

String String::num(double p_num, int p_decimals) {
	if (std::isnan(p_num)) {
		return "nan";
	}
	if (std::isinf(p_num)) {
		return p_num < 0 ? "-inf" : "inf";
	}
	char buf[64];
	if (p_decimals < 0) {
		snprintf(buf, sizeof(buf), "%.14g", p_num);
	} else {
		snprintf(buf, sizeof(buf), "%.*f", p_decimals > 16 ? 16 : p_decimals, p_num);
	}
	return String(buf);
}

String String::num_int64(int64_t p_num, int p_base, bool p_capitalize_hex) {
	char buf[72];
	if (p_base == 16) {
		const unsigned long long mag = p_num < 0 ? 0ull - (unsigned long long)p_num : (unsigned long long)p_num;
		snprintf(buf, sizeof(buf), p_capitalize_hex ? "%s%llX" : "%s%llx", p_num < 0 ? "-" : "", mag);
	} else {
		snprintf(buf, sizeof(buf), "%lld", (long long)p_num);
	}
	return String(buf);
}

String String::num_real(double p_num, bool p_trailing) {
	String s = num(p_num);
	if (p_trailing && s._s.find_first_of(".einf") == std::string::npos) {
		s._s += ".0";
	}
	return s;
}

String String::sprintf(const Array &p_values, bool *r_error) const {
	std::string out;
	int value_index = 0;
	bool error = false;
	const std::string &f = _s;
	for (size_t i = 0; i < f.size(); i++) {
		if (f[i] != '%') {
			out += f[i];
			continue;
		}
		if (i + 1 < f.size() && f[i + 1] == '%') {
			out += '%';
			i++;
			continue;
		}
		// %[flags][width][.precision]conversion
		size_t j = i + 1;
		std::string spec = "%";
		while (j < f.size() && strchr("-+ 0#", f[j])) {
			spec += f[j++];
		}
		while (j < f.size() && isdigit((unsigned char)f[j])) {
			spec += f[j++];
		}
		if (j < f.size() && f[j] == '.') {
			spec += f[j++];
			while (j < f.size() && isdigit((unsigned char)f[j])) {
				spec += f[j++];
			}
		}
		if (j >= f.size()) {
			error = true;
			break;
		}
		const char conv = f[j];
		if (value_index >= p_values.size()) {
			error = true;
			break;
		}
		const Variant &v = p_values[value_index++];
		char buf[512];
		switch (conv) {
			case 'd':
			case 'i':
				snprintf(buf, sizeof(buf), (spec + "lld").c_str(), (long long)v);
				out += buf;
				break;
			case 'x':
			case 'X':
			case 'o':
				snprintf(buf, sizeof(buf), (spec + "ll" + conv).c_str(), (unsigned long long)(long long)v);
				out += buf;
				break;
			case 'f':
			case 'F':
			case 'e':
			case 'E':
			case 'g':
			case 'G':
				snprintf(buf, sizeof(buf), (spec + conv).c_str(), double(v));
				out += buf;
				break;
			case 'c':
				_append_utf8(out, char32_t((long long)v));
				break;
			case 's':
			case 'v': {
				const String s = v.stringify();
				snprintf(buf, sizeof(buf), (spec + "s").c_str(), s.get_data());
				out += buf;
			} break;
			default:
				error = true;
				break;
		}
		if (error) {
			break;
		}
		i = j;
	}
	if (value_index != p_values.size()) {
		error = true;
	}
	if (r_error) {
		*r_error = error;
	}
	return error ? String("not all arguments converted during string formatting") : String(out);
}

String operator+(const char *p_a, const String &p_b) {
	return String(p_a) + p_b;
}

String operator+(char32_t p_a, const String &p_b) {
	return String() + p_a + p_b;
}

bool operator==(const char *p_a, const String &p_b) {
	return p_b == p_a;
}

bool operator!=(const char *p_a, const String &p_b) {
	return !(p_b == p_a);
}

String itos(int64_t p_val) {
	return String::num_int64(p_val);
}

String rtos(double p_val) {
	return String::num(p_val);
}

String vformat_array(const String &p_text, const Array &p_args) {
	bool error = false;
	const String fmt = p_text.sprintf(p_args, &error);
	ERR_FAIL_COND_V_MSG(error, String(), String("Formatting error in string \"") + p_text + "\": " + fmt + ".");
	return fmt;
}

// print_string.h

void __print_line(const String &p_string) {
	if (!CoreGlobals::print_line_enabled) {
		return;
	}
	fputs(p_string.get_data(), stdout);
	fputc('\n', stdout);
	fflush(stdout);
}

String stringify_variants(const Array &p_args) {
	String s;
	for (int i = 0; i < p_args.size(); i++) {
		if (i > 0) {
			s += " ";
		}
		s += p_args[i].stringify();
	}
	return s;
}

} // namespace gdl
