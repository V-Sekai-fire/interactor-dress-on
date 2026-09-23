#!/usr/bin/env bash
# Drive the loop over transport-godot-mcp, the way an agent would (Gate 0E's
# shape): play project/xr_main.tscn flat, wait for the in-game bridge by
# polling the PORT (Godot buffers a redirected stdout, so the log is no
# readiness signal), then call_method on /root/Main:
#   dress_on_stages            which stage ELFs are present, and why not
#   dress_on_run [ALLOW]       start the pipeline (returns at once)
#   dress_on_status            polled every 2 s until DONE or FAILED(...)
#   dress_on_result            the per-state record (JSON text)
# and stop the game. Requests and responses land in mcp/ beside this file.
#
#   gates/8-loop/mcp_run.sh                      # allow_fixture=infer,rig
#   ALLOW=infer,rig,curvenet,fit,drape gates/8-loop/mcp_run.sh
#
# MCP RESULT: PASS means the MCP chain drove the pipeline to a terminal state
# with every call answered (isError false). Whether that state is a loop
# PASS is Gate 8's verdict (gate_loop.gd), printed here as the final status.
# The port defaults to 8795 (not 8789) so a game another session has open
# does not answer instead; the runtime takes it from --mcp-port=.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
GODOT="${GODOT:-$HOME/scoop/apps/godot/current/godot.console.exe}"
PORT="${MCP_PORT:-8795}"
ALLOW="${ALLOW:-infer,rig}"
WALL="${MCP_WALL:-3600}"
URL="http://127.0.0.1:$PORT/mcp"
OUT="$HERE/mcp"
mkdir -p "$OUT"
rm -f "$OUT"/*.json

"$GODOT" --path "$ROOT/project" --rendering-driver vulkan --xr-mode off res://xr_main.tscn -- --mcp-port="$PORT" \
	> "$OUT/game.log" 2>&1 &
PID=$!

up=0
for i in $(seq 1 120); do
	if curl -s -o /dev/null -m 1 "$URL"; then up=1; break; fi
	sleep 0.5
done
if [ "$up" != 1 ]; then
	echo "FAIL: no MCP bridge on $PORT after 60 s"
	kill $PID 2>/dev/null
	exit 1
fi

n=0
rc=0
call() { # method args-json -> prints the response, keeps req/resp
	n=$((n + 1))
	printf '{"jsonrpc":"2.0","id":%s,"method":"tools/call","params":{"name":"call_method","arguments":{"path":"/root/Main","method":"%s","args":%s}}}' \
		"$n" "$1" "$2" > "$OUT/req_$n.json"
	curl -s -m 60 -H "Content-Type: application/json" -H "Accept: application/json, text/event-stream" \
		--data-binary @"$OUT/req_$n.json" "$URL" > "$OUT/resp_$n.json"
	grep -q '"isError":false' "$OUT/resp_$n.json" || rc=1
	cat "$OUT/resp_$n.json"
}
value() { # the call_method value out of a response (SSE: its data: line)
	grep -a '^data:' "$1" | sed 's/.*\\"value\\":\\"\(.*\)\\"}".*/\1/'
}

echo "--- dress_on_stages"; call dress_on_stages '[]'; echo
echo "--- dress_on_run [\"$ALLOW\"]"; call dress_on_run "[\"$ALLOW\"]"; echo
grep -q 'STARTED' "$OUT/resp_$n.json" || rc=1
t0=$(date +%s)
status=""
while :; do
	sleep 2
	call dress_on_status '[]' > /dev/null
	status="$(value "$OUT/resp_$n.json")"
	rm -f "$OUT/req_$n.json" "$OUT/resp_$n.json"
	n=$((n - 1))
	case "$status" in
		DONE*|FAILED*) break ;;
	esac
	if [ $(( $(date +%s) - t0 )) -gt "$WALL" ]; then
		echo "TIMEOUT after $WALL s in: $status"
		rc=1
		break
	fi
done
n=$((n + 1))
echo "{\"final_status\": \"$status\"}" > "$OUT/resp_${n}_status.json"
echo "--- final dress_on_status: $status"
echo "--- dress_on_result"; call dress_on_result '[]' | head -c 2000; echo

kill $PID 2>/dev/null
wait $PID 2>/dev/null
case "$status" in DONE*|FAILED*) ;; *) rc=1 ;; esac
echo "MCP RESULT: $([ $rc = 0 ] && echo PASS || echo FAIL) (pipeline: $status)"
exit $rc
