// Godot RenderingDevice enum values the guest needs, pinned by hand.
//
// The guest reaches RenderingDevice through Object::call with integer
// arguments, so the engine's enums have to be spelled out here. Every value
// below was read from Godot 4.7.2, servers/rendering/rendering_device_commons.h.
// Gate 0A learned the cost of a wrong one: 6 for STORAGE_BUFFER (that is
// IMAGE_BUFFER) fails only at uniform_set_create, as a validator message and a
// null RID. Add values here, with the enum they come from, never inline.
#pragma once

namespace rdc {

// RenderingDevice.ShaderStage: VERTEX=0 FRAGMENT=1 TESSELATION_CONTROL=2
// TESSELATION_EVALUATION=3 COMPUTE=4
constexpr int SHADER_STAGE_COMPUTE = 4;

// RenderingDevice.UniformType: SAMPLER=0 SAMPLER_WITH_TEXTURE=1 TEXTURE=2
// IMAGE=3 TEXTURE_BUFFER=4 SAMPLER_WITH_TEXTURE_BUFFER=5 IMAGE_BUFFER=6
// UNIFORM_BUFFER=7 STORAGE_BUFFER=8 INPUT_ATTACHMENT=9
constexpr int UNIFORM_TYPE_UNIFORM_BUFFER = 7;
constexpr int UNIFORM_TYPE_STORAGE_BUFFER = 8;

// @GlobalScope.Error: OK is 0. buffer_update returns one.
constexpr int GODOT_OK = 0;

} // namespace rdc
