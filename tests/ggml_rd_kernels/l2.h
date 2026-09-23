// L2 of the ggml-rd kernel checks: every kernel's slangc cpp emit, packed by
// the guest's own packers (guest/ggml-rd/rd_pack.cpp + ops/*.cpp), against
// native ggml-cpu, host-side, before anything runs on a GPU.
//
// A case builds a ggml graph in one context whose memory stands in for one
// RD buffer: every tensor's byte offset is its distance from the context
// base, exactly what the guest computes from its fake buffer base. The
// harness fills the leaves, computes the graph with ggml-cpu (the
// reference), restores the leaves, then packs and runs every non-layout
// node through its kernel's cpp emit and compares the output with the
// thresholds test-backend-ops uses (NMSE), counting bit-exact elements too.
//
// A family adds cases/<family>.cpp with an L2_CASES block, like
// cases/binary.cpp (ADD and MUL, the template); nothing else changes.
#pragma once

#include <functional>
#include <string>
#include <vector>

#include "ggml.h"

struct L2Case {
	std::string name;
	// Build the graph in ctx and return the node whose output is compared.
	std::function<ggml_tensor *(ggml_context *ctx)> build;
	float lo = -1.0f, hi = 1.0f; // leaves are uniform in [lo, hi)
	double max_nmse = 1e-7; // test-backend-ops' default for these ops
	// The output does not depend on the source values (SOFT_MAX of a one-element
	// row is 1): the swapped-stride control cannot change it, so counts it NOOP.
	bool value_blind = false;
};

using L2Maker = void (*)(std::vector<L2Case> &out);

struct L2Registrar {
	explicit L2Registrar(L2Maker m);
};

#define L2_CASES(ident)                                          \
	static void l2_cases_##ident(std::vector<L2Case> &out);      \
	static const L2Registrar l2_registrar_##ident(l2_cases_##ident); \
	static void l2_cases_##ident(std::vector<L2Case> &out)
