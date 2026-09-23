// The AVBD oracle in the guest, one-shot on the CPU backend. Returns "PASS cpu: ..."
// or "FAIL cpu: ..." with the numbers; the host wraps it for MCP. The rd
// oracle is the frame-driven job "fixture" (guest/drape/avbd_jobs.cpp): a
// one-shot rd call would sync in its submit's frame (AGENTS.md rule 4).
#pragma once

#include <string>

std::string avbd_fixture_cpu();
