"""Print the fixed evaluator's retention (scripts/retention_eval.py JSONs) for every run in a ret/ directory."""
import glob
import json
import os
import sys

for f in sorted(glob.glob(os.path.join(sys.argv[1], "*.json"))):
    d = json.load(open(f))
    n = os.path.basename(f)[:-5]
    cells = []
    for c, r in d["runs"].items():
        tag = os.path.basename(c).replace("_ckpt.pt", "").replace("ckpt_", "")
        last = " last %.3f" % r["old_head_last"] if "old_head_last" in r else ""
        cells.append("%s: old %.3f new %.3f%s" % (tag, r["old_head"], r["new_head"], last))
    print("%-22s expert %.3f | %s" % (n, d["old_expert"], " | ".join(cells)))
