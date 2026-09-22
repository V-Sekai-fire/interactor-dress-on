#!/usr/bin/env bash
# Drive the Stage 1 probes over transport-godot-mcp, the way an agent would.
#
# Plays project/ (flat, no XR), waits for the in-game bridge on 8789 by
# polling the PORT (Godot buffers stdout to a file, so the log is not a
# readiness signal), then issues tools/call -> call_method on /root/Main for
# rd_probe and rd_bench, and stops the game. Responses land beside this file.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
GODOT="${GODOT:-$HOME/scoop/apps/godot/current/godot.console.exe}"
PORT="${MCP_PORT:-8789}"
URL="http://127.0.0.1:$PORT/mcp"

"$GODOT" --path "$ROOT/project" --rendering-driver vulkan --xr-mode off > "$HERE/mcp_game.log" 2>&1 &
PID=$!

up=0
for i in $(seq 1 60); do
	if curl -s -o /dev/null -m 1 "$URL"; then up=1; break; fi
	sleep 0.5
done
if [ "$up" != 1 ]; then
	echo "FAIL: no MCP bridge on $PORT after 30 s"
	kill $PID 2>/dev/null
	exit 1
fi

call() { # id method args-json
	printf '{"jsonrpc":"2.0","id":%s,"method":"tools/call","params":{"name":"call_method","arguments":{"path":"/root/Main","method":"%s","args":%s}}}' "$1" "$2" "$3"
}

rc=0
call 1 rd_open '[]'               > "$HERE/mcp_req_1.json"
call 2 rd_probe '[]'              > "$HERE/mcp_req_2.json"
call 3 rd_bench '[64, 1, true]'   > "$HERE/mcp_req_3.json"
call 4 rd_bench '[1, 64, true]'   > "$HERE/mcp_req_4.json"
call 5 rd_close '[]'              > "$HERE/mcp_req_5.json"
for i in 1 2 3 4 5; do
	curl -s -m 30 -H "Content-Type: application/json" -H "Accept: application/json, text/event-stream" \
		--data-binary @"$HERE/mcp_req_$i.json" "$URL" > "$HERE/mcp_resp_$i.json"
	echo "--- $i: $(cat "$HERE/mcp_req_$i.json" | sed 's/.*"method":"\([a-z_]*\)".*/\1/')"
	cat "$HERE/mcp_resp_$i.json"; echo
	grep -q '"isError":false' "$HERE/mcp_resp_$i.json" || rc=1
done
grep -q 'PASS: GPU compute reached' "$HERE/mcp_resp_2.json" || rc=1
grep -q 'OK value=64 expected=64' "$HERE/mcp_resp_3.json" || rc=1
grep -q 'OK value=64 expected=64' "$HERE/mcp_resp_4.json" || rc=1

kill $PID 2>/dev/null
wait $PID 2>/dev/null
rm -f "$HERE"/mcp_req_*.json
echo "MCP RESULT: $([ $rc = 0 ] && echo PASS || echo FAIL)"
exit $rc
