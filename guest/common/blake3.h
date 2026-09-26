// blake3 -- BLAKE3 hash mode, streaming, std types only.
//
// The one checksum the gates compare, on both sides of the boundary: a guest
// hashes what it read or computed, the host (Python's blake3 package, or the
// guest again for bytes that crossed) hashes the same bytes, and the gates
// compare the first 12 hex digits. Only for equality checks; nothing here is
// secret. Picked over SHA-256 for speed: 7 rounds of 32-bit add/xor/rotate
// on 64-byte blocks, no message schedule, and a cryptographic hash still.
//
// This is the reference implementation's structure (the chunk state plus a
// stack of chaining values, merged as chunks complete), portable and
// single-threaded; no SIMD paths, which a RISC-V guest has no use for.
// Header-only so any stage can include it without a library.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>

namespace blake3 {

namespace detail {

constexpr uint32_t IV[8] = { 0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
	0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19 };
constexpr uint8_t PERM[16] = { 2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8 };
enum : uint32_t {
	CHUNK_START = 1,
	CHUNK_END = 2,
	PARENT = 4,
	ROOT = 8,
};
constexpr size_t BLOCK = 64, CHUNK = 1024;

inline uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

inline void g(uint32_t *s, int a, int b, int c, int d, uint32_t mx, uint32_t my) {
	s[a] = s[a] + s[b] + mx;
	s[d] = rotr(s[d] ^ s[a], 16);
	s[c] = s[c] + s[d];
	s[b] = rotr(s[b] ^ s[c], 12);
	s[a] = s[a] + s[b] + my;
	s[d] = rotr(s[d] ^ s[a], 8);
	s[c] = s[c] + s[d];
	s[b] = rotr(s[b] ^ s[c], 7);
}

// The compression function; out gets all 16 words (the first 8 are the
// chaining value).
inline void compress(const uint32_t cv[8], const uint32_t block[16], uint64_t counter, uint32_t len,
		uint32_t flags, uint32_t out[16]) {
	uint32_t s[16] = { cv[0], cv[1], cv[2], cv[3], cv[4], cv[5], cv[6], cv[7], IV[0], IV[1], IV[2], IV[3],
		uint32_t(counter), uint32_t(counter >> 32), len, flags };
	uint32_t m[16];
	std::memcpy(m, block, sizeof m);
	for (int r = 0; r < 7; r++) {
		g(s, 0, 4, 8, 12, m[0], m[1]);
		g(s, 1, 5, 9, 13, m[2], m[3]);
		g(s, 2, 6, 10, 14, m[4], m[5]);
		g(s, 3, 7, 11, 15, m[6], m[7]);
		g(s, 0, 5, 10, 15, m[8], m[9]);
		g(s, 1, 6, 11, 12, m[10], m[11]);
		g(s, 2, 7, 8, 13, m[12], m[13]);
		g(s, 3, 4, 9, 14, m[14], m[15]);
		if (r < 6) {
			uint32_t t[16];
			for (int i = 0; i < 16; i++)
				t[i] = m[PERM[i]];
			std::memcpy(m, t, sizeof m);
		}
	}
	for (int i = 0; i < 8; i++) {
		out[i] = s[i] ^ s[i + 8];
		out[i + 8] = s[i + 8] ^ cv[i];
	}
}

inline void words(const uint8_t *b, uint32_t w[16]) {
	for (int i = 0; i < 16; i++)
		w[i] = uint32_t(b[4 * i]) | uint32_t(b[4 * i + 1]) << 8 | uint32_t(b[4 * i + 2]) << 16 |
				uint32_t(b[4 * i + 3]) << 24;
}

// What a chunk or parent node would compress to, kept until we know whether
// it is the root.
struct Output {
	uint32_t cv[8];
	uint32_t block[16];
	uint64_t counter;
	uint32_t len, flags;
	void chaining_value(uint32_t out[8]) const {
		uint32_t o[16];
		compress(cv, block, counter, len, flags, o);
		std::memcpy(out, o, 8 * sizeof(uint32_t));
	}
};

inline Output parent(const uint32_t l[8], const uint32_t r[8]) {
	Output o;
	std::memcpy(o.cv, IV, sizeof o.cv);
	std::memcpy(o.block, l, 8 * sizeof(uint32_t));
	std::memcpy(o.block + 8, r, 8 * sizeof(uint32_t));
	o.counter = 0;
	o.len = BLOCK;
	o.flags = PARENT;
	return o;
}

} // namespace detail

class Ctx {
public:
	Ctx() { reset(); }

	void reset() {
		std::memcpy(cv_, detail::IV, sizeof cv_);
		chunk_ = 0;
		buf_len_ = 0;
		blocks_ = 0;
		stack_len_ = 0;
	}

	void update(const void *data, size_t n) {
		const uint8_t *p = static_cast<const uint8_t *>(data);
		while (n > 0) {
			// a full chunk is only finished once more input shows it is not the last
			if (chunk_bytes() == detail::CHUNK) {
				uint32_t cv[8];
				chunk_output().chaining_value(cv);
				push_chunk(cv, ++chunk_);
				std::memcpy(cv_, detail::IV, sizeof cv_);
				blocks_ = 0;
				buf_len_ = 0;
			}
			if (buf_len_ == detail::BLOCK) {
				uint32_t w[16], o[16];
				detail::words(buf_, w);
				detail::compress(cv_, w, chunk_, detail::BLOCK, start_flag(), o);
				std::memcpy(cv_, o, sizeof cv_);
				blocks_++;
				buf_len_ = 0;
			}
			size_t want = detail::CHUNK - chunk_bytes();
			size_t room = detail::BLOCK - buf_len_;
			size_t take = n < room ? n : room;
			if (take > want)
				take = want;
			std::memcpy(buf_ + buf_len_, p, take);
			buf_len_ += take;
			p += take;
			n -= take;
		}
	}

	// 32-byte digest as 64 lowercase hex digits.
	std::string hex() const {
		detail::Output o = chunk_output();
		for (size_t i = stack_len_; i-- > 0;) {
			uint32_t cv[8];
			o.chaining_value(cv);
			o = detail::parent(stack_[i], cv);
		}
		uint32_t out[16];
		detail::compress(o.cv, o.block, o.counter, o.len, o.flags | detail::ROOT, out);
		static const char *d = "0123456789abcdef";
		std::string s(64, '0');
		for (int i = 0; i < 32; i++) {
			const uint8_t b = uint8_t(out[i / 4] >> (8 * (i % 4)));
			s[2 * i] = d[b >> 4];
			s[2 * i + 1] = d[b & 15];
		}
		return s;
	}

private:
	size_t chunk_bytes() const { return blocks_ * detail::BLOCK + buf_len_; }
	uint32_t start_flag() const { return blocks_ == 0 ? detail::CHUNK_START : 0; }

	detail::Output chunk_output() const {
		detail::Output o;
		std::memcpy(o.cv, cv_, sizeof o.cv);
		uint8_t b[detail::BLOCK] = {};
		std::memcpy(b, buf_, buf_len_);
		detail::words(b, o.block);
		o.counter = chunk_;
		o.len = uint32_t(buf_len_);
		o.flags = start_flag() | detail::CHUNK_END;
		return o;
	}

	// Merge completed subtrees: one merge per trailing zero bit of the total.
	void push_chunk(uint32_t cv[8], uint64_t total) {
		while ((total & 1) == 0) {
			detail::parent(stack_[--stack_len_], cv).chaining_value(cv);
			total >>= 1;
		}
		std::memcpy(stack_[stack_len_++], cv, 8 * sizeof(uint32_t));
	}

	uint32_t cv_[8];
	uint64_t chunk_; // index of the chunk being filled
	uint8_t buf_[detail::BLOCK];
	size_t buf_len_;
	size_t blocks_; // blocks already compressed in this chunk
	uint32_t stack_[54][8];
	size_t stack_len_;
};

inline std::string hex(const void *data, size_t n) {
	Ctx c;
	c.update(data, n);
	return c.hex();
}

} // namespace blake3
