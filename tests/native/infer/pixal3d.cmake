# ── Pixal3D (Stage 7): proj attention + the 7a pipeline over trellis2 ──────
# Included by CMakeLists.txt after the trellis2 block. Both executables link
# the trellis2 library (vendor/trellis2, Aero-Ex loader + proj mode).
#   pixal3d-7a          image -> DINOv3 -> SS flow (proj) -> SS dec -> MC -> OBJ
#   pixal3d-proj-block  one-block forward on fixed inputs, for the torch oracle
#                       gates/7-pixal3d/proj/proj_block_ref.py
add_executable(pixal3d-7a ${CMAKE_CURRENT_LIST_DIR}/pixal3d_7a.cpp)
target_include_directories(pixal3d-7a PRIVATE ${T2}/examples ${T2}/stb)
target_link_libraries(pixal3d-7a PRIVATE trellis2)
add_executable(pixal3d-proj-block ${CMAKE_CURRENT_LIST_DIR}/pixal3d_proj_block.cpp)
target_link_libraries(pixal3d-proj-block PRIVATE trellis2)
