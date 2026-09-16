#!/bin/bash
# Every mutation introduces ONE real defect. The suite MUST turn red for each.
# Usage: tests/mutate.sh .
SRC="$1"; PY=/Users/flaviacasini/claude-bot/venv/bin/python3
run_mutation() {
  local nome="$1"; local py_patch="$2"
  local d; d=$(mktemp -d)
  cp -r "$SRC"/wattwright "$SRC"/tests "$d"/ 2>/dev/null
  $PY - "$d/wattwright" <<PYEOF
import sys
p = sys.argv[1]; s = open(p).read()
$py_patch
open(p, "w").write(s)
PYEOF
  if $PY "$d/tests/test_wattwright.py" >/dev/null 2>&1; then
    echo "  ✗ NOT CAUGHT    — $nome"
  else
    echo "  ✓ caught       — $nome"
  fi
  rm -rf "$d"
}

run_mutation "temperature check deleted from measure_point" \
  's = s.replace("""        hot = too_hot(s)
        if hot is not None:
            raise RuntimeError(f\"ABORT: GPU{hot} reached {TEMP_ABORT} C\")
        reads.append(s)""", "        reads.append(s)")'

run_mutation "clock restore emptied out" \
  's = s.replace("errors = unlock_all(self.indexes)", "errors = []")'

run_mutation "rollback of a partial lock removed" \
  's = s.replace("""        if mhz is not None:                       # roll the partial lock back
            for i in done:
                _lock_one(i, None)""", "        pass")'

run_mutation "stop() no longer waits for the in-flight request" \
  's = s.replace("            t.join(timeout)", "            pass")'

run_mutation "reading speed returns 0.0 instead of None" \
  's = s.replace("""        n = (d.get(\"usage\") or {}).get(\"prompt_tokens\")
        if isinstance(n, (int, float)) and n > 0 and d.get(\"_elapsed\"):
            return float(n) / d[\"_elapsed\"]
        return None""", "        return 0.0")'

run_mutation "[N/A] becomes 0.0 again" \
  's = s.replace("""    except ValueError:
        return None""", """    except ValueError:
        return 0.0""")'

run_mutation "percentage rule compared naively instead of rounded" \
  's = s.replace("""              if round(p[\"tps\"] / reference[\"tps\"] * 100) >= round(ratio * 100)]""", """              if p[\"tps\"] / reference[\"tps\"] >= ratio]""")'

run_mutation "install power stops removing the existing unit" \
  's = s.replace("            os.remove(UNIT_PATH)", "            pass")'

run_mutation "systemctl failure ignored" \
  's = s.replace("    if not systemctl(\"daemon-reload\") or not systemctl(\"enable\", \"--now\", \"wattwright\"):", "    if False:")'

run_mutation "power invented with no unlocked run in the file" \
  's = s.replace("""    if unlocked is not None:
        prof[\"power\"] = unlocked""", "    prof[\"power\"] = reference")'

run_mutation "points with no power reading accepted anyway" \
  's = s.replace("""    missing = sorted(set(indexes) - set(per_gpu))
    if missing:""", """    missing = []
    if missing:""")'
