// The AVBD oracle in the guest, for both backends. Each returns "PASS <backend>: ..."
// or "FAIL <backend>: ..." with the numbers; the host wraps them for MCP.
#pragma once

#include <string>

#include "rd_compute.h"

std::string avbd_fixture_cpu();
std::string avbd_fixture_rd(rdc::Device &dev);
