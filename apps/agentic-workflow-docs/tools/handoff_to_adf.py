#!/usr/bin/env python3
"""Convert a HANDOFF.md-style ticket body into Jira ADF and emit an
acli `edit --from-json` envelope. Reusable for every app's Jira task.

Usage: handoff_to_adf.py <HANDOFF.md> <ISSUE_KEY> <out.json> [tracking.json]
tracking.json (optional) = list of "Key: value" strings for the info panel.
"""
import sys, json, re, itertools

_ids = itertools.count(1)
def lid(prefix):  # ADF requires localId on task nodes
    return f"{prefix}-{next(_ids)}"

def txt(s):
    return {"type": "text", "text": s}

def para(s):
    s = s.strip()
    return {"type": "paragraph", "content": [txt(s)]} if s else None

def heading(s, level=2):
    return {"type": "heading", "attrs": {"level": level}, "content": [txt(s.strip())]}

def bullet_list(items):
    items = [i.strip() for i in items if i.strip()]
    return {
        "type": "bulletList",
        "content": [
            {"type": "listItem", "content": [{"type": "paragraph", "content": [txt(i)]}]}
            for i in items
        ],
    }

def task_list(items):  # items = list of (state, text)
    return {
        "type": "taskList",
        "attrs": {"localId": lid("tl")},
        "content": [
            {"type": "taskItem",
             "attrs": {"localId": lid("ti"), "state": state},
             "content": [txt(text.strip())]}
            for state, text in items if text.strip()
        ],
    }

def parse_tasks(lines):
    items, cur = [], None
    for ln in lines:
        m = re.match(r'^\s*\[([ xX])\]\s+(.*)$', ln)
        if m:
            if cur:
                items.append(cur)
            cur = ["DONE" if m.group(1).lower() == "x" else "TODO", m.group(2)]
        elif ln.strip() == "":
            continue
        elif cur:  # wrapped continuation of the current task
            cur[1] += " " + ln.strip()
    if cur:
        items.append(cur)
    return [task_list(items)] if items else []

def parse_mixed(lines):
    """Prose paragraphs (reflowed) interleaved with bullet runs."""
    nodes, pbuf, bullets = [], [], None
    def flush_p():
        nonlocal pbuf
        if pbuf:
            t = " ".join(x.strip() for x in pbuf if x.strip())
            n = para(t)
            if n:
                nodes.append(n)
            pbuf = []
    def flush_b():
        nonlocal bullets
        if bullets:
            nodes.append(bullet_list(bullets))
        bullets = None
    for ln in lines:
        if re.match(r'^\s*[-*]\s+', ln):
            flush_p()
            if bullets is None:
                bullets = []
            bullets.append(re.sub(r'^\s*[-*]\s+', '', ln).strip())
        elif ln.strip() == "":
            flush_p(); flush_b()
        elif bullets is not None and (ln.startswith("  ") or ln.startswith("\t")):
            bullets[-1] += " " + ln.strip()  # wrapped bullet continuation
        else:
            flush_b()
            pbuf.append(ln)
    flush_p(); flush_b()
    return nodes

# Known section headers (prefix match on a stripped line)
SECTIONS = ["Goal", "Todo List", "GATE 3 validation results",
            "Tier B log", "Tier C runtime-risk checklist",
            "Deliverables", "References"]

def is_header(line):
    s = line.strip()
    if s.startswith(("-", "[", "*")) or len(s) > 90:
        return None
    for p in SECTIONS:
        if s == p or s.startswith(p):
            return s
    return None

def main():
    handoff, key, out = sys.argv[1], sys.argv[2], sys.argv[3]
    tracking = json.load(open(sys.argv[4])) if len(sys.argv) > 4 else []
    lines = open(handoff, encoding="utf-8").read().splitlines()

    content = []

    # Tracking info panel (visually distinct at the top)
    if tracking:
        content.append({
            "type": "panel", "attrs": {"panelType": "info"},
            "content": [bullet_list(tracking),
                        para("This ticket mirrors the app's HANDOFF.md "
                             "(one source of truth: HANDOFF.md == PR body == this Jira task).")],
        })

    # Slice HANDOFF into (header, body-lines) blocks
    blocks, hdr, buf = [], None, []
    for ln in lines:
        h = is_header(ln)
        if h:
            if hdr is not None:
                blocks.append((hdr, buf))
            hdr, buf = h, []
        else:
            if hdr is None:
                continue
            buf.append(ln)
    if hdr is not None:
        blocks.append((hdr, buf))

    for hdr, body in blocks:
        content.append(heading(hdr, 2))
        if hdr.startswith("Todo List"):
            content.extend(parse_tasks(body))
        else:
            content.extend(parse_mixed(body))

    content = [c for c in content if c]
    doc = {"type": "doc", "version": 1, "content": content}
    json.dump({"issues": [key], "description": doc}, open(out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"wrote {out}: {len(content)} top-level nodes")

if __name__ == "__main__":
    main()
