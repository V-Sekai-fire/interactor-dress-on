import LeanSlang
import Ggml.SlangCodegen.Common
import Ggml.SlangCodegen.Binary
import Ggml.SlangCodegen.Unary
import Ggml.SlangCodegen.Rope
import Ggml.SlangCodegen.Cpy
import Ggml.SlangCodegen.GetRows
import Ggml.SlangCodegen.Concat
import Ggml.SlangCodegen.Repeat
import Ggml.SlangCodegen.Upscale
import Ggml.SlangCodegen.Norm
import Ggml.SlangCodegen.SoftMax
import Ggml.SlangCodegen.MulMatTiled
import Ggml.SlangCodegen.MulMatVec
import Ggml.SlangCodegen.MulMatSerial
import Ggml.SlangCodegen.Conv
import Ggml.SlangCodegen.FlashAttn
import Ggml.SlangCodegen.UnarySeeThrough
import Ggml.SlangCodegen.Glu
import Ggml.SlangCodegen.GroupNorm
import Ggml.SlangCodegen.Pad
import Ggml.SlangCodegen.Arange
import Ggml.SlangCodegen.TimestepEmbedding
import Ggml.SlangCodegen.Conv2d
import Ggml.SlangCodegen.ConvTranspose2d

/-!
# `Ggml` — the ggml-rd op kernels (Lean → Slang → spirv | cpp)

Each family module under `Ggml.SlangCodegen` builds its kernels from
`Common` (the fixed layout and the params words) and exports
`kernels : List (String × SlangShaderModule)`. `Ggml.kernels` below is the
concatenation `lake exe emit_ggml` writes; `kernels/ggml/kernels.txt`
picks which of them the guest embeds. A family adds one import line and
one `++` line here.
-/

namespace Ggml

def kernels : List (String × LeanSlang.SlangShaderModule) :=
  Ggml.SlangCodegen.Binary.kernels
  ++ Ggml.SlangCodegen.Unary.kernels
  ++ Ggml.SlangCodegen.Rope.kernels
  ++ Ggml.SlangCodegen.Cpy.kernels
  ++ Ggml.SlangCodegen.GetRows.kernels
  ++ Ggml.SlangCodegen.Concat.kernels
  ++ Ggml.SlangCodegen.Repeat.kernels
  ++ Ggml.SlangCodegen.Upscale.kernels
  ++ Ggml.SlangCodegen.Norm.kernels
  ++ Ggml.SlangCodegen.SoftMax.kernels
  ++ Ggml.SlangCodegen.MulMatTiled.kernels ++ Ggml.SlangCodegen.MulMatVec.kernels
  ++ Ggml.SlangCodegen.MulMatSerial.kernels
  ++ Ggml.SlangCodegen.Conv.kernels
  ++ Ggml.SlangCodegen.FlashAttn.kernels
  ++ Ggml.SlangCodegen.UnarySeeThrough.kernels
  ++ Ggml.SlangCodegen.Glu.kernels
  ++ Ggml.SlangCodegen.GroupNorm.kernels
  ++ Ggml.SlangCodegen.Pad.kernels
  ++ Ggml.SlangCodegen.Arange.kernels
  ++ Ggml.SlangCodegen.TimestepEmbedding.kernels
  ++ Ggml.SlangCodegen.Conv2d.kernels
  ++ Ggml.SlangCodegen.ConvTranspose2d.kernels

/-- Control kernels: deliberately off the fixed layout, for gates only
    (kernels/ggml/controls.txt). -/
def controls : List (String × LeanSlang.SlangShaderModule) :=
  Ggml.SlangCodegen.Binary.controls

end Ggml
