cmake_minimum_required(VERSION 3.16)
project(threadprobe CXX)

# The sandbox-api cmake helper comes from the house demo. Gate 0A deliberately
# depends on nothing else: if this gate needs more than godot-sandbox to build,
# the independence constraint is already lost.
if(NOT DEFINED SANDBOX_DEMO_ROOT)
	set(SANDBOX_DEMO_ROOT "C:/contract-manifest/3-interactor/mujoco-sandbox-demo")
endif()

add_subdirectory("${SANDBOX_DEMO_ROOT}/third_party/sandbox-api/cmake" sandbox_cmake)

add_sandbox_program(threadprobe guest/main.cpp)
target_link_libraries(threadprobe PRIVATE -lm -lstdc++)
