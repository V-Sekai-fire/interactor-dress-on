// Gate 0G: see usd_probe_core.h. No file is ever opened: plugInfo.json comes
// through the Plug_InMemoryPlugInfoHook patch (gates/0g-openusd/patch_usd.py),
// generatedSchema.usda and the input bytes through UsdProbe_MemResolver
// (guest/usd/mem_resolver.cpp since Cut U, where usd.elf shares it).

#include "usd_probe_core.h"

#include "mem_resolver.h"

#include "pxr/pxr.h"
#include "pxr/base/tf/errorMark.h"
#include "pxr/usd/sdf/layer.h"
#include "pxr/usd/usd/primRange.h"
#include "pxr/usd/usd/stage.h"
#include "pxr/usd/usdGeom/mesh.h"
#include "pxr/usd/usdSkel/root.h"
#include "pxr/usd/usdSkel/skeleton.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdp {

static std::string first_error(const TfErrorMark &m) {
	for (auto it = m.GetBegin(); it != m.GetEnd(); ++it)
		return it->GetCommentary();
	return "unknown";
}

std::string init() {
	return usdmem::init();
}

static void fnv(uint64_t &h, const void *p, size_t n) {
	const unsigned char *b = static_cast<const unsigned char *>(p);
	for (size_t i = 0; i < n; ++i) {
		h ^= b[i];
		h *= 1099511628211ull;
	}
}

std::string load(const std::string &bytes, int path_mode) {
	init();
	TfErrorMark mark;
	const bool crate = bytes.size() >= 8 && std::memcmp(bytes.data(), "PXR-USDC", 8) == 0;
	const char *fmt = crate ? "usdc" : "usda";
	UsdStageRefPtr stage;
	try {
		if (!crate && path_mode == 0) {
			SdfLayerRefPtr layer = SdfLayer::CreateAnonymous(".usda");
			if (!layer->ImportFromString(bytes))
				return "ERR: fmt=usda ImportFromString failed: " + first_error(mark);
			stage = UsdStage::Open(layer, UsdStage::LoadAll);
		} else {
			// A fresh path per load: the layer registry never hands back a
			// cached layer for new bytes.
			static int serial = 0;
			std::string path = "/mem/input" + std::to_string(++serial) + "." + fmt;
			usdmem::put_asset(path, bytes.data(), bytes.size());
			SdfLayerRefPtr layer = SdfLayer::FindOrOpen(path);
			if (!layer)
				return std::string("ERR: fmt=") + fmt + " open failed: " + first_error(mark);
			stage = UsdStage::Open(layer, UsdStage::LoadAll);
		}
	} catch (const std::exception &e) {
		return std::string("ERR: fmt=") + fmt + " exception: " + e.what();
	}
	if (!stage)
		return std::string("ERR: fmt=") + fmt + " no stage: " + first_error(mark);

	size_t prims = 0, meshes = 0, skels = 0, roots = 0, npts = 0, nfvi = 0;
	uint64_t h = 1469598103934665603ull;
	for (const UsdPrim &prim : stage->Traverse()) {
		++prims;
		if (prim.IsA<UsdSkelSkeleton>())
			++skels;
		if (prim.IsA<UsdSkelRoot>())
			++roots;
		if (!prim.IsA<UsdGeomMesh>())
			continue;
		++meshes;
		UsdGeomMesh mesh(prim);
		VtVec3fArray pts;
		VtIntArray fvi;
		mesh.GetPointsAttr().Get(&pts, UsdTimeCode::Default());
		mesh.GetFaceVertexIndicesAttr().Get(&fvi, UsdTimeCode::Default());
		npts += pts.size();
		nfvi += fvi.size();
		fnv(h, prim.GetPath().GetString().data(), prim.GetPath().GetString().size());
		if (!pts.empty())
			fnv(h, pts.cdata(), pts.size() * sizeof(GfVec3f));
		if (!fvi.empty())
			fnv(h, fvi.cdata(), fvi.size() * sizeof(int));
	}
	char buf[256];
	std::snprintf(buf, sizeof buf, "ok fmt=%s prims=%zu meshes=%zu skels=%zu skelroots=%zu points=%zu fvi=%zu cksum=%016llx",
			fmt, prims, meshes, skels, roots, npts, nfvi, (unsigned long long)h);
	std::string out = buf;
	if (!mark.IsClean())
		out += " warn=" + first_error(mark);
	return out;
}

} // namespace usdp
