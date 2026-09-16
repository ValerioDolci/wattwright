#!/usr/bin/env python3
"""Mutation testing: break one guarantee at a time and check the suite notices.

Every entry below introduces ONE real defect that a past review actually found.
The suite must turn red for each. Three outcomes are possible and they mean
different things:

    caught      the tests spotted the defect - the guarantee is guarded
    NOT CAUGHT  the defect passed unnoticed - the tests do not cover it
    STALE       the patch no longer matches the source, so nothing was mutated

STALE matters as much as NOT CAUGHT. A mutation whose target text has drifted
applies nothing, the suite passes for the wrong reason, and a dead check starts
looking like a live one. That happened here once, which is why it is detected.

Passing all of these means these specific regressions are guarded. It does not
mean the code is free of bugs.

    python3 tests/mutate.py [path-to-repo]
"""
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

# (name, text to find, text to replace it with)
MUTATIONS: list[tuple[str, str, str]] = [
    ("temperature check deleted from measure_point",
     '        hot = too_hot(s)\n        if hot is not None:\n'
     '            raise RuntimeError(f"ABORT: GPU{hot} reached {TEMP_ABORT} C")\n'
     '        reads.append(s)',
     '        reads.append(s)'),

    ("clock restore emptied out",
     "errors = unlock_all(self.indexes)", "errors = []"),

    ("rollback of a partial lock removed",
     "            back = [e for e in (_lock_one(i, None) for i in done) if e]",
     "            back = []  # "),

    ("stop() no longer waits for the in-flight request",
     "            t.join(timeout)", "            pass"),

    ("reading speed returns 0.0 instead of None",
     '        n = (d.get("usage") or {}).get("prompt_tokens")\n'
     '        if isinstance(n, (int, float)) and n > 0 and d.get("_elapsed"):\n'
     '            return float(n) / d["_elapsed"]\n        return None',
     "        return 0.0"),

    ("[N/A] becomes 0.0 again",
     "    except ValueError:\n        return None", "    except ValueError:\n        return 0.0"),

    ("percentage rule compared naively instead of rounded",
     '              if round(p["tps"] / reference["tps"] * 100) >= round(ratio * 100)]',
     '              if p["tps"] / reference["tps"] >= ratio]'),

    ("install power stops removing the existing unit",
     "            os.remove(UNIT_PATH)", "            pass"),

    ("systemctl failure ignored",
     '    if not systemctl("daemon-reload") or not systemctl("enable", "--now", "wattwright"):',
     "    if False:"),

    ("power invented with no unlocked run in the file",
     '    if unlocked is not None:\n        prof["power"] = unlocked',
     '    prof["power"] = reference'),

    ("points with no power reading accepted anyway",
     "    missing = sorted(set(indexes) - set(per_gpu))\n    if missing:",
     "    missing = []\n    if missing:"),

    # --- second round: silence, lies and NaN ------------------------------
    ("missing telemetry treated as a cool GPU again",
     '    absent = sorted(set(indexes) - {g["i"] for g in read})\n    if absent:\n'
     '        raise RuntimeError(f"no telemetry from GPU {absent}: is nvidia-smi failing?")',
     "    pass"),

    ("failed rollback no longer reported",
     '            if back:\n                errors.append("AND the rollback failed: "',
     '            if False:\n                errors.append("AND the rollback failed: "'),

    ("a restarted load thread inherits the old error",
     "        with self._lock:\n            self._error = None", "        if False:\n            pass"),

    ("NaN and Infinity accepted as measurements",
     '            and x == x and x not in (float("inf"), float("-inf")))', "            )"),

    ("points that are not a list crash again",
     "    if not isinstance(raw, list):\n        return []", "    raw = raw or []"),

    ("the first free run wins instead of the fastest",
     '    unlocked = max(free_runs, key=lambda p: p["tps"]) if free_runs else None',
     "    unlocked = free_runs[0] if free_runs else None"),

    ("table groups by clock and merges different points",
     "        groups.setdefault(id(p), []).append(name)",
     '        groups.setdefault(p.get("clock"), []).append(name)'),

    ("a point with no tokens/s is recorded anyway",
     '                if point["tps"] is None:', "                if False:"),
]


def main() -> int:
    repo = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, ".."))
    source = open(os.path.join(repo, "wattwright")).read()
    caught = stale = escaped = 0

    for name, find, replace in MUTATIONS:
        if find not in source:
            print(f"  ! STALE      - {name}")
            stale += 1
            continue
        with tempfile.TemporaryDirectory() as d:
            shutil.copytree(os.path.join(repo, "tests"), os.path.join(d, "tests"))
            with open(os.path.join(d, "wattwright"), "w") as f:
                f.write(source.replace(find, replace, 1))
            r = subprocess.run([PY, os.path.join(d, "tests", "test_wattwright.py")],
                               capture_output=True, text=True)
        if r.returncode == 0:
            print(f"  x NOT CAUGHT - {name}")
            escaped += 1
        else:
            print(f"  v caught     - {name}")
            caught += 1

    print(f"\n{caught} caught, {escaped} not caught, {stale} stale "
          f"(out of {len(MUTATIONS)})")
    return 0 if (escaped == 0 and stale == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
