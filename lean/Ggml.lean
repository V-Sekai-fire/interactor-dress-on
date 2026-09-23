import LeanSlang
import Ggml.SlangCodegen.Common
import Ggml.SlangCodegen.Binary
import Ggml.SlangCodegen.Unary
import Ggml.SlangCodegen.Rope
import Ggml.SlangCodegen.Cpy
import Ggml.SlangCodegen.GetRows
import Ggml.SlangCodegen.Concat
import Ggml.SlangCodegen.Repeat
import Ggml.SlangCodegen.Norm
import Ggml.SlangCodegen.SoftMax

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
  ++ Ggml.SlangCodegen.Norm.kernels
  ++ Ggml.SlangCodegen.SoftMax.kernels

/-- Control kernels: deliberately off the fixed layout, for gates only
    (kernels/ggml/controls.txt). -/
def controls : List (String × LeanSlang.SlangShaderModule) :=
  Ggml.SlangCodegen.Binary.controls

end Ggml
