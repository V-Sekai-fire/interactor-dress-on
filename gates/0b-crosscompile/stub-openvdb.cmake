# Gate 0B: neutralise OpenVDB so configure can proceed and report what ELSE
# fails. cloth-fit's recipe is `if(TARGET openvdb::openvdb) return()`, and
# polyfem links the bare name `openvdb`. Pre-defining both makes the recipe a
# no-op. FitForm.cpp (the one consumer) will fail to compile -- expected; the
# plan replaces that SDF grid rather than porting OpenVDB. Everything else
# compiling or not is the gate's real answer.
#
# Guarded, because CMAKE_PROJECT_INCLUDE_BEFORE runs on EVERY project() call,
# including each CPM subproject's -- the first version of this file collided
# with itself inside polysolve.
if(NOT TARGET openvdb)
	add_library(openvdb INTERFACE)
	add_library(openvdb::openvdb ALIAS openvdb)
endif()
