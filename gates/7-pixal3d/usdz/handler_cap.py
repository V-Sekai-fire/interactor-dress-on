"""Control for the RunPod handler's result cap, on the desk (no Docker, no GPU).

  pixi run -m tools/services/pixal3d python gates/7-pixal3d/usdz/handler_cap.py > gates/7-pixal3d/usdz/handler_cap.log

Imports tools/runpod/pixal3d/handler.py with a stand-in `runpod` module (the SDK is only
in the linux-64 environment; handler() never calls it), lets it start the real serve.py
in WEFTSPUN_STUB mode, then sends an extract job twice: under the default 20 MB cap it
must come back as the service's JSON (format usdz), and with the cap lowered below the
stub package's size it must come back as {"error": ...}, which RunPod reports as a
failed job, instead of a result the gateway would drop.
"""
import os
import socket
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.modules["runpod"] = types.ModuleType("runpod")  # handler imports it at module level
s = socket.socket()
s.bind(("127.0.0.1", 0))
os.environ["PIXAL3D_PORT"] = str(s.getsockname()[1])
s.close()
os.environ["WEFTSPUN_STUB"] = "1"
sys.path.insert(0, str(ROOT / "tools" / "runpod" / "pixal3d"))
import handler  # noqa: E402


def main() -> int:
    handler.start_server()
    try:
        ok = True
        job = {"input": {"route": "extract", "state": "c3R1YnN0YXRl"}}
        out = handler.handler(job)
        good = out.get("format") == "usdz" and "usd_b64" in out and "error" not in out
        print(f"{'ok  ' if good else 'FAIL'} cap {handler.RESULT_CAP} B: keys {sorted(out)}, "
              f"usd_b64 {len(out.get('usd_b64', ''))} B")
        ok = ok and good
        handler.RESULT_CAP = 1000
        out = handler.handler(job)
        good = "cap" in out.get("error", "")
        print(f"{'ok  ' if good else 'FAIL'} cap {handler.RESULT_CAP} B: {out}")
        ok = ok and good
        print("PASS" if ok else "FAIL")
        return 0 if ok else 1
    finally:
        handler._stop_server()


if __name__ == "__main__":
    raise SystemExit(main())
