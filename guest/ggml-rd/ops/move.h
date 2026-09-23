// ggml-rd data-movement ops (cpy.cpp, concat.cpp, repeat.cpp, get_rows.cpp):
// the shared packing. The kernels are lean/Ggml/SlangCodegen/{Move,Cpy,
// GetRows,Concat,Repeat}.lean; Move.lean says how a kernel stores a 16-bit
// element (the thread at a word's even address writes the word) and what it
// needs from the packer, which is here:
//
//   iteration order  dst's dimensions sorted by element stride, size-1
//                    dimensions last (stable), so consecutive threads write
//                    consecutive addresses and "the next address" is a carry
//                    over the sorted dimensions;
//   nested           the sorted layout has nb[k+1] >= ne[k] * nb[k]: no two
//                    elements share an address and no dimension interleaves
//                    another. A 16-bit destination is supported only then;
//   merge chain      word 55: bit 0 when nb[0] = 1, bit k when bit k-1 is
//                    and nb[k] = ne[k-1] * nb[k-1] (over the non-trivial
//                    dimensions).
//
// The packer permutes the dst block and every source block whose dimensions
// follow dst's (CONCAT, REPEAT) by the same order; a CPY keeps src0's block
// and gets dst's original linear weights in iteration order (words 56-59).
#pragma once

#include <cstdint>
#include <initializer_list>

#include "../rd_pack.h"

namespace ggml_rd {
namespace move {

constexpr uint32_t W_MERGE = 55; // Move.wMerge
constexpr uint32_t W_LINW = 56; // Move.wLinW 0

struct Order {
	int perm[4]; // iteration dim k is original dim perm[k]
	uint64_t ne[4], nb[4]; // in iteration order, nb in elements
	int nontrivial; // dims with ne > 1 (they come first)
};

inline Order order_of(const int64_t ne[4], const uint64_t nb_elems[4]) {
	Order o;
	int p[4] = { 0, 1, 2, 3 };
	auto key_less = [&](int a, int b) {
		const bool ta = ne[a] == 1, tb = ne[b] == 1;
		if (ta != tb) {
			return !ta; // non-trivial first
		}
		if (ta) {
			return false; // keep the original order of size-1 dims
		}
		return nb_elems[a] < nb_elems[b];
	};
	// Stable insertion sort of four.
	for (int i = 1; i < 4; ++i) {
		const int x = p[i];
		int j = i - 1;
		while (j >= 0 && key_less(x, p[j])) {
			p[j + 1] = p[j];
			--j;
		}
		p[j + 1] = x;
	}
	o.nontrivial = 0;
	for (int k = 0; k < 4; ++k) {
		o.perm[k] = p[k];
		o.ne[k] = uint64_t(ne[p[k]]);
		o.nb[k] = nb_elems[p[k]];
		o.nontrivial += ne[p[k]] > 1;
	}
	return o;
}

inline Order order_of(const ggml_tensor *t) {
	const size_t ts = ggml_type_size(t->type);
	uint64_t nb[4];
	for (int k = 0; k < 4; ++k) {
		nb[k] = uint64_t(t->nb[k]) / ts;
	}
	return order_of(t->ne, nb);
}

inline bool nested(const Order &o) {
	for (int k = 0; k < o.nontrivial; ++k) {
		if (o.nb[k] == 0) {
			return false;
		}
		if (k > 0 && o.nb[k] < o.ne[k - 1] * o.nb[k - 1]) {
			return false;
		}
	}
	return true;
}

inline uint32_t merge_chain(const Order &o) {
	uint32_t m = 0;
	for (int k = 0; k < o.nontrivial; ++k) {
		const bool link = k == 0 ? o.nb[0] == 1 : o.nb[k] == o.ne[k - 1] * o.nb[k - 1];
		if (!link) {
			break;
		}
		m |= 1u << k;
	}
	return m;
}

inline bool is16(ggml_type t) {
	return ggml_type_size(t) == 2;
}

// Permute one tensor block's ne and nb words by the iteration order.
inline void permute_block(uint32_t *w, uint32_t at, const Order &o) {
	uint32_t ne[4], nb[4];
	for (int k = 0; k < 4; ++k) {
		ne[k] = w[at + T_NE + o.perm[k]];
		nb[k] = w[at + T_NB + o.perm[k]];
	}
	for (int k = 0; k < 4; ++k) {
		w[at + T_NE + k] = ne[k];
		w[at + T_NB + k] = nb[k];
	}
}

// The iteration order of the dst block already in the slot (fill_standard
// wrote it in elements), applied to dst and to the listed source blocks;
// word 55 gets the merge chain. False if a 16-bit dst is not nested.
inline bool apply_order(Pack &p, Order &o, std::initializer_list<uint32_t> src_blocks) {
	int64_t ne[4];
	uint64_t nb[4];
	for (int k = 0; k < 4; ++k) {
		ne[k] = p.w[W_DST + T_NE + k];
		nb[k] = p.w[W_DST + T_NB + k];
	}
	o = order_of(ne, nb);
	if (is16(p.node->type) && !nested(o)) {
		return false;
	}
	permute_block(p.w, W_DST, o);
	for (uint32_t b : src_blocks) {
		permute_block(p.w, b, o);
	}
	p.w[W_MERGE] = merge_chain(o);
	return true;
}

// A 16-bit destination must be nested (the kernels' word ownership).
inline bool dst_ok(const ggml_tensor *dst) {
	return !is16(dst->type) || nested(order_of(dst));
}

} // namespace move
} // namespace ggml_rd
