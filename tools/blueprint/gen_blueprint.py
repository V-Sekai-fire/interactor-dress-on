#!/usr/bin/env python3
"""Turn a markdown table (issue #5's step table) into a blueprint node series.

Reads the first pipe table in a markdown file (columns: Step, What they use,
What we have, State) and writes Unreal Engine blueprint clipboard text that
blueprintUE's render.js (tools/blueprint/bue-render, from the org's
blueprint-self-hosted) draws: an event node, then one function node per row,
their exec pins chained in table order, each row's other columns as input
pins with their text as the default value. Rows are laid out left to right,
wrapping every `--per-row` nodes.

    python3 tools/blueprint/gen_blueprint.py issue.md > series.txt
    python3 tools/blueprint/gen_blueprint.py issue.md --html out.html

With --html it writes a self-contained page next to bue-render/ (the page
loads bue-render/render.css and render.js relative to itself).
"""
import argparse
import html
import re
import sys
import uuid


def table_rows(text):
    rows = [l.strip() for l in text.splitlines() if l.strip().startswith('|')]
    cells = [[c.strip() for c in r.strip('|').split('|')] for r in rows]
    cells = [c for c in cells if not all(re.fullmatch(r':?-{3,}:?', x) for x in c)]
    return cells[0], cells[1:]


def plain(s):
    s = re.sub(r'`([^`]*)`', r'\1', s)
    s = re.sub(r'\*\*([^*]*)\*\*', r'\1', s)
    return s.replace('"', "'").replace('\\', '/')


def guid(seed):
    return uuid.uuid5(uuid.NAMESPACE_URL, 'dress-on-blueprint/' + seed).hex.upper()


def node_width(row, header):
    # render.js sizes a node to its longest pin (label + default value); ~6.3 px
    # per character at zoom 1:1, plus the pin label and margins.
    return max(len(plain(h)) + len(plain(v)) for h, v in zip(header[1:], row[1:])) * 6.3 + 120


def series(header, rows, per_row, gap=80, dy=420, title='Series'):
    out = []
    xs, x = [], 0
    for i, row in enumerate(rows):
        if i % per_row == 0:
            x = 520
        xs.append(x)
        x += node_width(row, header) + gap
    ev = 'K2Node_CustomEvent_0'
    ev_out = guid('event/then')
    out.append('Begin Object Class=/Script/BlueprintGraph.K2Node_CustomEvent Name="%s"' % ev)
    out.append('   CustomFunctionName="%s"' % plain(title))
    out.append('   NodePosX=0')
    out.append('   NodePosY=0')
    out.append('   NodeGuid=%s' % guid('event'))
    out.append('   CustomProperties Pin (PinId=%s,PinName="then",Direction="EGPD_Output",'
               'PinType.PinCategory="exec",LinkedTo=(K2Node_CallFunction_0 %s,))' % (ev_out, guid('0/execute')))
    out.append('End Object')
    for i, row in enumerate(rows):
        name = 'K2Node_CallFunction_%d' % i
        line = i // per_row
        out.append('Begin Object Class=/Script/BlueprintGraph.K2Node_CallFunction Name="%s"' % name)
        out.append('   FunctionReference=(MemberName="%s")' % plain(row[0]))
        out.append('   NodePosX=%d' % xs[i])
        out.append('   NodePosY=%d' % (line * dy))
        out.append('   NodeGuid=%s' % guid('%d' % i))
        prev = (ev, ev_out) if i == 0 else ('K2Node_CallFunction_%d' % (i - 1), guid('%d/then' % (i - 1)))
        out.append('   CustomProperties Pin (PinId=%s,PinName="execute",Direction="EGPD_Input",'
                   'PinType.PinCategory="exec",LinkedTo=(%s %s,))' % (guid('%d/execute' % i), prev[0], prev[1]))
        nxt = ',LinkedTo=(K2Node_CallFunction_%d %s,)' % (i + 1, guid('%d/execute' % (i + 1))) if i + 1 < len(rows) else ''
        out.append('   CustomProperties Pin (PinId=%s,PinName="then",Direction="EGPD_Output",'
                   'PinType.PinCategory="exec"%s)' % (guid('%d/then' % i), nxt))
        for h, v in zip(header[1:], row[1:]):
            out.append('   CustomProperties Pin (PinId=%s,PinName="%s",Direction="EGPD_Input",'
                       'PinType.PinCategory="string",DefaultValue="%s")' % (guid('%d/%s' % (i, h)), plain(h), plain(v)))
        out.append('End Object')
    return '\n'.join(out) + '\n'


PAGE = '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>%(title)s</title>
<link href="bue-render/render.css" rel="stylesheet">
<style>html,body{margin:0;background:#1a1c1f}.hidden{display:none}</style>
</head>
<body>
<div class="playground"></div>
<textarea class="hidden" id="pastebin_data">%(data)s</textarea>
<script src="bue-render/render.js"></script>
<script>
new window.blueprintUE.render.Main(
  document.getElementById('pastebin_data').value,
  document.getElementsByClassName('playground')[0],
  {height: window.innerHeight + 'px'}
).start();
</script>
</body>
</html>
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('markdown')
    ap.add_argument('--per-row', type=int, default=5)
    ap.add_argument('--title', default='Astra workflow (issue 5)')
    ap.add_argument('--html')
    a = ap.parse_args()
    header, rows = table_rows(open(a.markdown, encoding='utf-8').read())
    text = series(header, rows, a.per_row, title=a.title)
    if a.html:
        open(a.html, 'w', encoding='utf-8').write(PAGE % {'title': html.escape(a.title), 'data': html.escape(text)})
        print('wrote %s: %d nodes' % (a.html, len(rows) + 1))
    else:
        sys.stdout.write(text)


if __name__ == '__main__':
    main()
