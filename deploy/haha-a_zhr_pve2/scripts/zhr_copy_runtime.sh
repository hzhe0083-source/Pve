#!/usr/bin/env bash
# Our own copy of the lab's Isaac runtime, so the a2r experiment is independent of the lab's files (user's choice A,
# 2026-10-31): the RoboDojo conda env and the three source trees it installs in editable mode (Isaac Lab, cuRobo,
# XPolicyLab) are copied read-only into $E, and only the copy's editable finders are pointed at the copied sources.
# The lab's Kit logs are left out. Nothing under the lab's directories is written. Run on haha-c:
#   bash scripts/zhr_copy_runtime.sh > ~/zhr_pve/logs/copy_runtime.log 2>&1
set -euo pipefail
SRC_ENV=/home/zhaobo/anaconda3/envs/RoboDojo
SRC_R=/home/zhaobo/agentic-robodojo/RoboDojo
E=/home/zhaobo/zhr_pve/a2r_env
SP=$E/RoboDojo/lib/python3.11/site-packages
echo "COPY_START $(date +%T)"
mkdir -p "$E/src" "$E/home"
cp_ro() { ionice -c3 nice -n19 rsync -a "$@"; }
cp_ro --exclude 'lib/python3.11/site-packages/isaacsim/kit/logs/' "$SRC_ENV/" "$E/RoboDojo/"
cp_ro "$SRC_R/third_party/IsaacLab/" "$E/src/IsaacLab/"
cp_ro "$SRC_R/third_party/curobo/" "$E/src/curobo/"
cp_ro "$SRC_R/XPolicyLab/" "$E/src/XPolicyLab/"
mkdir -p "$SP/isaacsim/kit/logs"
echo "COPIED $(date +%T) $(du -sh "$E" | cut -f1)"

# point the copy's editable installs at the copied sources (our files only)
sed -i -e "s#$SRC_R/third_party/IsaacLab#$E/src/IsaacLab#g" -e "s#$SRC_R/third_party/curobo#$E/src/curobo#g" \
    -e "s#$SRC_R/XPolicyLab#$E/src/XPolicyLab#g" "$SP"/__editable___*_finder.py "$SP"/*.dist-info/direct_url.json
left=$(grep -l "agentic-robodojo" "$SP"/__editable___*_finder.py || true)
[ -z "$left" ] || { echo "UNREWRITTEN $left"; exit 1; }
cmp -s "$SRC_ENV/bin/python3.11" "$E/RoboDojo/bin/python3.11" || { echo "PYTHON_MISMATCH"; exit 1; }

# every package resolves to our copies, checked with find_spec: importing isaaclab_tasks needs pxr, which exists only
# once the Kit app runs, and omni.* is provided by Kit at runtime (HOME is ours, so Kit caches never land in the lab's
# ~/.cache/ov)
HOME=$E/home "$E/RoboDojo/bin/python" -I - <<'PY'
import importlib.util, sys
want = "/home/zhaobo/zhr_pve/a2r_env/"
print("python", sys.version.split()[0], sys.prefix)
for name in ("isaaclab", "isaaclab_tasks", "isaaclab_assets", "isaaclab_rl", "isaaclab_mimic", "isaaclab_contrib",
             "curobo", "XPolicyLab", "isaacsim", "torch"):
    s = importlib.util.find_spec(name)
    loc = s.origin if s.origin not in (None, "namespace") else list(s.submodule_search_locations)[0]
    print("%-16s %s" % (name, loc))
    assert loc.startswith(want), "%s resolves outside our copy: %s" % (name, loc)
PY
echo "COPY_OK $(date +%T)"
