"""wattwright tests. All offline: no GPU, no `nvidia-smi`, no endpoint.

What matters here is not that the code runs - it is that the profile RULES are the
ones written in the README even on awkward curves: flat, proportional, missing data.
A profile deriver that is quietly wrong tells someone to run their machine at the
wrong clock for months.
"""
import contextlib, importlib.machinery, importlib.util, io, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_loader(
    "ww", importlib.machinery.SourceFileLoader("ww", os.path.join(HERE, "..", "wattwright")))
ww = importlib.util.module_from_spec(spec); spec.loader.exec_module(ww)

failed = 0


def check(cond, msg):
    global failed
    if not cond:
        failed += 1
        print("FAIL:", msg)


def pt(clock, tps, watt, fan=50, temp=65):
    return {"clock": clock, "tps": tps, "watt_total": watt, "fan_mean": fan,
            "temp_mean": temp, "prefill_tps": tps * 16}


# -- 1. the real curve of a 2x RTX 5070 Ti box: the rules must find the right points
real = {"points": [pt(None, 123.3, 505, 67), pt(2700, 123.4, 465, 63),
                   pt(2400, 117.17, 367, 52), pt(2100, 105.1, 303, 44),
                   pt(1800, 92.7, 272, 47), pt(1500, 79.5, 241, 40)]}
p = ww.derive(real)
check(p["power"]["clock"] is None, "power must be the free clock")
check(p["free"]["clock"] == 2700, f"free should be 2700, got {p['free']['clock']}")
check(p["quiet"]["clock"] == 2400, f"quiet should be 2400, got {p['quiet']['clock']}")
check(p["eco"]["clock"] == 2100, f"eco should be 2100, got {p['eco']['clock']}")
# The real numbers put 2400 at 94.997% of the free run: a naive `>= 0.95` rejects
# it while the table prints "95%". The rule compares rounded percentages, as shown.
check(round(117.17 / 123.3 * 100) == 95, "the table would print 95% for this point")

# -- 2. FLAT curve (nothing costs anything): free must go all the way down
flat = {"points": [pt(None, 100, 500), pt(2400, 100, 400), pt(1500, 100, 300)]}
p = ww.derive(flat)
check(p["free"]["clock"] == 1500, "on a flat curve free must take the lowest clock")
check(p["quiet"]["clock"] == 1500, "quiet lands there too: both names stay valid")
check(p["eco"]["clock"] == 1500, "and so does eco")

# -- 3. PROPORTIONAL curve (compute-bound): nothing is free
prop = {"points": [pt(None, 120, 480), pt(2400, 96, 384), pt(1800, 72, 288)]}
p = ww.derive(prop)
check("free" not in p, "on a proportional curve there is no free point")
check("quiet" not in p, "nor a point costing 5% or less")
check(p["eco"]["clock"] in (2400, 1800), "eco must still exist (best tokens per watt)")

# -- 4. degenerate inputs: invent nothing
check(ww.derive({"points": []}) == {}, "no points, no profiles")
check(ww.derive({"points": [pt(None, 0, 0)]}) == {}, "a point without tokens/s is unusable")
only_free = ww.derive({"points": [pt(None, 100, 500)]})
check(list(only_free) == ["power"], f"with only the free run there is only power: {list(only_free)}")

# -- 5. eco picks the efficiency peak, not simply the lowest clock
eff = {"points": [pt(None, 100, 500), pt(2400, 95, 300), pt(1500, 60, 290)]}
p = ww.derive(eff)
check(p["eco"]["clock"] == 2400, "eco took the lowest clock instead of the most efficient one")
check(p["quiet"]["clock"] == 2400, "quiet lands on the same point and must stay addressable")

# -- 6. resolve(): names, raw numbers, and refusing what does not exist
check(ww.resolve(real, "power") is None, "power means no lock")
check(ww.resolve({}, "2200") == 2200, "a raw MHz value works without measurements")
check(ww.resolve(real, "eco") == 2100, "eco must resolve to the measured clock")
try:
    ww.resolve(real, "turbo")
    check(False, "an unknown profile should have raised")
except SystemExit:
    pass

# -- 7. the table groups names landing on the same clock without losing any
ww.load_data = lambda _: flat
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    ww.cmd_profiles(type("A", (), {"data": None})())
table = buf.getvalue()
check("free / quiet / eco" in table, f"the three names should share one row:\n{table}")
check(len([r for r in table.splitlines() if "1500" in r]) == 1,
      "one row for that clock, not three")

# -- 8. the boot unit: right commands per GPU, and it knows how to undo itself
ww.gpu_indexes = lambda: [0, 1]
ww.shutil.which = lambda _: "/usr/bin/nvidia-smi"
ww.load_data = lambda _: real
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    ww.cmd_install(type("A", (), {"profile": "eco", "data": None, "show": True})())
unit = buf.getvalue()
check(unit.count("ExecStart=") == 2, "one ExecStart per GPU")
check("-lgc 2100" in unit, "the unit must lock at eco's clock")
check(unit.count("ExecStop=") == 2 and "-rgc" in unit, "the unit must be able to restore")
check("RemainAfterExit=yes" in unit, "without RemainAfterExit systemd calls it failed")

print("FAILED:", failed) if failed else print("all ok")
sys.exit(1 if failed else 0)
