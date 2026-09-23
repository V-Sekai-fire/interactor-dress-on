// curvenet smoke, Geogram half: no godot_lite prelude in this TU (Geogram's
// and gdl's unqualified names are kept apart, as in the ELF's libraries).
#include "delaunay_geogram.h"

#include <geogram/basic/common.h>
#include <geogram/basic/logger.h>
#include <geogram/delaunay/delaunay.h>

#include <cstdio>

// Returns the number of 2D Delaunay cells Geogram's BDEL2d makes of four
// points, or -1 if the factory did not hand back a BDEL2d (the fork had
// dropped its registration once; see vendor/geogram-subset/CITATION.cff).
int smoke_geogram_bdel2d(const double *p_xy, int p_n) {
	GEO::initialize(GEO::GEOGRAM_INSTALL_NONE);
	GEO::Logger::instance()->set_quiet(true);
	GEO::Delaunay_var d = GEO::Delaunay::create(2, "BDEL2d");
	if (d.get() == nullptr) {
		return -1;
	}
	d->set_vertices(GEO::index_t(p_n), p_xy);
	return int(d->nb_cells());
}

// The same four points through Cassie's wrapper (mwt's only Delaunay).
int smoke_cassie_delaunay_2d(const double *p_xy, int p_n) {
	int *faces = nullptr;
	int nf = 0;
	if (!cassie::delaunay_triangulate_2d_raw(p_xy, p_n, &faces, &nf)) {
		return 0;
	}
	delete[] faces;
	return nf;
}
