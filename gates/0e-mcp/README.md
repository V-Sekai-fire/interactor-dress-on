# Gate 0E — drive the RenderingDevice probe over transport-godot-mcp

**Result: PASS.**

The runtime bridge (`mcp_runtime.gd`, autoloaded, port 8789) answered 2 s
after the game started. One MCP `tools/call`:

```json
{"jsonrpc":"2.0","id":1,"method":"tools/call",
 "params":{"name":"call_method",
           "arguments":{"path":"/root/Main","method":"rd_probe","args":[]}}}
```

returned:

```json
{"id":1.0,"jsonrpc":"2.0",
 "result":{"content":[{"text":"{\"value\":\"PASS: GPU compute reached from the guest\"}",
                       "type":"text"}],"isError":false}}
```

So the chain is proved end to end: an MCP client → `mcp_runtime.gd` →
`Main.rd_probe()` → `Sandbox.vmcall("rd_probe", spirv)` → the riscv64 guest →
`RenderingDevice` compute on the GPU → and back. Stock Godot 4.7.2, the
`godot_sandbox` addon, the `vsekai_godot_mcp` addon, nothing else. This is
how every later stage gets exercised by an agent.

`req.json` and `response.json` beside this file are the exact bytes.

## The host wrapper

`project/main.gd` owns the Sandbox and exposes **no-argument** methods
(`rd_probe`, `rd_last_step`). The guest API takes typed buffers such as
`PackedByteArray`; wrapping keeps those off the JSON wire, and `call_method`
then needs no argument marshalling at all. Follow the same shape for the pen
and dress-on entry points.

## What cost time, so it does not again

- **Godot buffers stdout when redirected to a file.** The first run polled the
  game log for `[godot_mcp] RUNTIME MCP` and never saw it in 40 s, then killed
  the game; the line only flushed at exit. The `curl` got an empty reply
  because the game was killed, not because MCP was down. **Poll the port,
  not the log** — `curl` until `/mcp` answers with any HTTP status.
- **`--xr-mode off` for non-VR gates.** `project.godot` enables OpenXR for
  the VR stages; with no HMD present that adds an init stall and a page of
  loader errors before the scene starts. Nothing in 0E needs XR.
- The godot_sandbox editor plugin probes `ninja --version` and `zig --help`
  on import and logs an error when they are absent. Harmless.
