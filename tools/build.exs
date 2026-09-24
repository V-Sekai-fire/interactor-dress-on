# One step, the same on a desk and in CI: elixir tools/build.exs [options]
#
# It checks the tools, gets the riscv64 sysroot, cross-builds the guest ELFs
# into project/ (build.sh), builds and runs the host harnesses (the ggml-rd
# kernel L2, the G3.graph oracle), and, when Godot is on the PATH, imports the
# project, bakes the native translations of the ELFs into project/bintr/ and
# runs the headless gates. Every step prints what it runs; the first failure
# stops the build with a non-zero exit. Plain Elixir, no Mix, no dependencies:
# what erlef/setup-beam gives a GitHub runner and `apt install elixir` a desk.
#
#   --targets=a,b     the ELF targets to build (default: all but fit.elf)
#   --fit             also fit.elf (cloth-fit: the org forks, ~40 min)
#   --no-elfs         skip the cross-build (use the committed ELFs)
#   --no-host         skip the host harnesses
#   --no-bintr        skip the native translations
#   --gates=a,b       headless gates to run: curvenet,ggml_rd,wrappers (default: wrappers)
#   --sysroot=<dir>   the riscv64 sysroot (else $RISCV64_SYSROOT, else fetched)
#   --jobs=N          build parallelism (default: the machine's cores)
#
# Environment it honours: RISCV64_SYSROOT, BUILD_DIR (default build/rv64),
# SLANGC, SPIRV_VAL, CC (the compiler for the translations, default cc).
defmodule Build do
  @root Path.expand("..", __DIR__)
  @sysroot_repo "https://github.com/V-Sekai-fire/interactor-mujoco-sandbox-demo"
  @sysroot_sub "third_party/riscv64-sysroot"
  @elfs ~w(dress_on drape curvenet probes ggml_test rd_worker)

  def main(argv) do
    opts = parse(argv)
    say("checkout #{@root}")
    tools(opts)
    sysroot = sysroot(opts)
    if opts.elfs, do: elfs(opts, sysroot)
    if opts.host, do: host(opts)
    if System.find_executable("godot") do
      import_project()
      if opts.bintr, do: bintr(opts)
      gates(opts)
    else
      say("godot: not on PATH; the import, the translations and the gates are skipped")
    end
    say("done")
  end

  # --- options -----------------------------------------------------------------

  defp parse(argv) do
    {kv, _, _} =
      OptionParser.parse(argv,
        switches: [targets: :string, fit: :boolean, no_elfs: :boolean, no_host: :boolean, no_bintr: :boolean,
                   gates: :string, sysroot: :string, jobs: :integer])
    targets = if kv[:targets], do: String.split(kv[:targets], ","), else: @elfs
    targets = if kv[:fit], do: targets ++ ["fit"], else: targets
    %{
      targets: targets,
      fit: kv[:fit] || false,
      elfs: !kv[:no_elfs],
      host: !kv[:no_host],
      bintr: !kv[:no_bintr],
      gates: String.split(kv[:gates] || "wrappers", ",", trim: true),
      sysroot: kv[:sysroot] || System.get_env("RISCV64_SYSROOT"),
      jobs: kv[:jobs] || System.schedulers_online()
    }
  end

  # --- steps ---------------------------------------------------------------------

  defp tools(opts) do
    for t <- ~w(cmake ninja clang++ ld.lld python3 git), do: need(t)
    unless System.get_env("SLANGC"), do: need("slangc")
    unless System.get_env("SPIRV_VAL"), do: need("spirv-val")
    {targets, 0} = System.cmd("clang++", ["--print-targets"])
    unless targets =~ "riscv64", do: fail("clang++ has no riscv64 target")
    if opts.fit, do: need("pixi")
    say("tools: ok (#{opts.jobs} jobs)")
  end

  defp sysroot(%{sysroot: dir}) when is_binary(dir) do
    unless File.exists?(Path.join(dir, "toolchain.cmake")), do: fail("no toolchain.cmake under #{dir}")
    say("sysroot: #{dir}")
    dir
  end

  defp sysroot(_) do
    dir = Path.join([@root, "build", "riscv64-sysroot-src"])
    unless File.exists?(Path.join([dir, @sysroot_sub, "toolchain.cmake"])) do
      File.mkdir_p!(Path.dirname(dir))
      File.rm_rf!(dir)
      run("git", ~w(clone -q --depth 1 --filter=blob:none --sparse #{@sysroot_repo} #{dir}))
      run("git", ~w(-C #{dir} sparse-checkout set #{@sysroot_sub}))
    end
    sysroot = Path.join(dir, @sysroot_sub)
    say("sysroot: #{sysroot} (fetched from the org's mujoco demo)")
    sysroot
  end

  defp elfs(opts, sysroot) do
    env = [
      {"RISCV64_SYSROOT", sysroot},
      {"BUILD_DIR", System.get_env("BUILD_DIR") || Path.join([@root, "build", "rv64"])},
      {"BUILD_FIT", if(opts.fit, do: "1", else: "0")},
      {"BUILD_TARGETS", Enum.join(opts.targets, " ")},
      {"BUILD_JOBS", to_string(opts.jobs)}
    ]
    run("bash", [Path.join(@root, "build.sh")], env)
    # This machine's slangc may have rewritten the cpp emits with an absolute
    # include of its prelude; put the inline form back (tools/inline_prelude.py).
    run("python3", [Path.join(@root, "tools/inline_prelude.py"), @root])
    for t <- opts.targets, do: File.exists?(Path.join([@root, "project", "#{t}.elf"])) || fail("no project/#{t}.elf")
    say("elfs: #{Enum.join(opts.targets, " ")}")
  end

  defp host(opts) do
    l2 = Path.join([@root, "build", "l2"])
    cmake(Path.join(@root, "tests/ggml_rd_kernels"), l2, [], opts)
    run(Path.join(l2, "ggml_rd_l2"), [])
    run(Path.join(l2, "ggml_rd_l2"), ["--control=swap-nb"])
    oracle = Path.join([@root, "build", "oracle"])
    vulkan = if System.get_env("VULKAN_SDK") || System.find_executable("glslc"), do: "ON", else: "OFF"
    cmake(Path.join(@root, "tests/ggml_graph_oracle"), oracle, ["-DORACLE_VULKAN=#{vulkan}"], opts)
    say("host: L2 and its control PASS; oracle built (ggml-vulkan #{vulkan})")
  end

  defp import_project do
    run("godot", ~w(--path #{Path.join(@root, "project")} --headless --xr-mode off --import), [], allow_fail: true)
  end

  # Native translations: each ELF's translation as C (the addon writes it on
  # GODOT_SANDBOX_BINTR_EMIT), compiled into project/bintr/bintr-<HASH>.so,
  # which the addon loads when sandbox/binary_translation/enabled is on.
  defp bintr(_opts) do
    out = Path.join([@root, "project", "bintr"])
    File.mkdir_p!(out)
    run("godot", ~w(--path #{Path.join(@root, "project")} --headless --xr-mode off --script tools/bintr_emit.gd),
      [{"GODOT_SANDBOX_BINTR_EMIT", out}])
    sources = Path.wildcard(Path.join(out, "bintr-*.c"))
    if sources == [], do: fail("the addon wrote no translation (does it carry the emit patch?)")
    cc = System.get_env("CC") || "cc"
    for src <- sources do
      # The emitted file carries its own #defines (the flags it was translated
      # with, the host arch); the compile line is libriscv's own.
      hash = src |> Path.basename(".c") |> String.replace_prefix("bintr-", "") |> String.upcase()
      so = Path.join(out, "bintr-#{hash}.so")
      run(cc, ~w(-O2 -s -std=c99 -fPIC -shared -x c -fexceptions -fvisibility=hidden -fomit-frame-pointer) ++
        [src, "-o", so])
      File.rm!(src)
      say("bintr: #{Path.basename(so)}")
    end
  end

  defp gates(opts) do
    project = Path.join(@root, "project")
    for g <- opts.gates do
      script =
        case g do
          "wrappers" -> "tests/probe_main_wrappers.gd"
          "curvenet" -> "gate_curvenet.gd"
          "ggml_rd" -> "gate_ggml_rd.gd"
          other -> fail("unknown gate #{other}")
        end
      run("godot", ~w(--path #{project} --headless --xr-mode off --script #{script}))
    end
  end

  # --- helpers -------------------------------------------------------------------

  defp cmake(src, dir, extra, opts) do
    unless File.exists?(Path.join(dir, "build.ninja")) do
      run("cmake", ~w(-S #{src} -B #{dir} -G Ninja -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
                     -DCMAKE_BUILD_TYPE=Release) ++ extra)
    end
    run("cmake", ~w(--build #{dir} -- -j#{opts.jobs}))
  end

  defp run(cmd, args, env \\ [], o \\ []) do
    say("$ #{cmd} #{Enum.join(args, " ")}")
    {_, rc} = System.cmd(cmd, args, env: env, into: IO.stream(:stdio, :line), stderr_to_stdout: true, cd: @root)
    if rc != 0 and not Keyword.get(o, :allow_fail, false), do: fail("#{cmd} exited #{rc}")
    rc
  end

  defp need(tool), do: System.find_executable(tool) || fail("#{tool} is not on PATH")
  defp say(msg), do: IO.puts("== #{msg}")

  defp fail(msg) do
    IO.puts(:stderr, "build: #{msg}")
    System.halt(1)
  end
end

Build.main(System.argv())
