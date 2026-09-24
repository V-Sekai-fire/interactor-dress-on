# Parked 2026-09-24

Session outputs that lived only in a scratch directory, kept here when the
session was parked. The anonymous survey's text is not kept (counts only are in
issue 5).

- `cassie_labels.tsv`: the 89 CASSIE sketches in `V-Sekai/datasource-cassie`
  (raw_data JSON: sketchSystem, sketchModel, interactionMode) with stroke
  counts: 72 study sketches (12 per system x object), 17 free creations
  (among them a dress of 153 strokes and two hats).
- `astra-board.html`: the issue 5 board artifact as of its last publish (it
  predates the Skateboard's xr-grid fork and the title change).
- `issue-5-*.png`: the blueprint renders of issue 5 (PR #8, closed).

Open work at park time:
- PR #4 (ANNY kernels): green, a draft; mark ready to merge.
- PR #6 (macOS Metal CI): 021d4bc carries the Gate 3 fixes (reopened Sandbox
  after a killed vmcall, Metal NOTEs, GELU census rows cut); its CI run is the
  test. PR #9 is stacked on it (f6913e3 has it merged in).
- PR #10: AGENTS rule 1 exception for interactor-meshing-pen.
- interactor-meshing-pen PR #1 (feat/dress-on): the dress-on loop hosted in
  xr-grid, tools/test.sh (zero-shot load test), and the OXRSys replay gate
  (tools/replay_oxrsys.py, tools/gate_replay.gd, .github/workflows/macos.yml);
  its first macOS run is the test.
- CI runs each push twice (push and pull_request): restrict push to main and
  add a concurrency group to halve the queue.
- udon2godot: the pre-rebase main is kept as backup/pre-rebase-main.
- fix/main-authors: main with the author rewrite, to force-push by hand.
