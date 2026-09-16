"""wattwright tests. All offline: no GPU, no `nvidia-smi`, no endpoint.

What matters here is not that the code runs. It is that the two things which can
actually hurt somebody keep working: the clock always goes back to free, and a
card that gets too hot stops the measurement. An earlier version of this file
tested only the profile maths - the pure, side-effect-free part - and an audit
showed that deleting the temperature check or emptying the restore left the whole
suite green. Everything below exists because of that.

Run: python3 tests/test_wattwright.py
"""
import contextlib
import importlib.machinery
import importlib.util
import io
import json
import os
import signal
import sys
import tempfile
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_loader(
    "ww", importlib.machinery.SourceFileLoader("ww", os.path.join(HERE, "..", "wattwright")))
ww = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ww)

failed = 0
LOAD_DATA = ww.load_data      # the real one: several tests replace it with a stub


def check(cond, msg):
    global failed
    if not cond:
        failed += 1
        print("FAIL:", msg)


def pt(clock, tps, watt, fan=50, temp=65, prefill=None):
    return {"clock": clock, "tps": tps, "watt_total": watt, "fan_mean": fan,
            "temp_mean": temp, "prefill_tps": prefill if prefill is not None else tps * 16}


# The numbers actually measured on a 2x RTX 5070 Ti box. Kept EXACT on purpose:
# rounding 123.34 to 123.3 makes the 95% rule pass even when it is implemented
# wrongly, which is how a previous version of this test was blind to a real bug.
REAL = {"points": [pt(None, 123.34, 504.7, 67), pt(2700, 123.35, 465.0, 63),
                   pt(2400, 117.17, 366.8, 52), pt(2100, 105.11, 302.7, 44),
                   pt(1800, 92.74, 271.7, 47), pt(1500, 79.49, 240.8, 40)]}


# ══════════════════════════ 1. safety: the clock always comes back ══════════

class FakeSmi:
    """Stands in for nvidia-smi. Records every lock/unlock and can fail on cue."""

    def __init__(self, fail_on=(), fail_unlock_on=()):
        self.fail_on, self.fail_unlock_on = set(fail_on), set(fail_unlock_on)
        self.locked: dict[int, int | None] = {}
        self.calls: list[tuple[int, int | None]] = []

    def run(self, argv, capture_output=True, text=True):
        class R:
            returncode, stdout, stderr = 0, "", ""
        r = R()
        if argv[0] != "nvidia-smi":
            return r
        idx = int(argv[argv.index("-i") + 1])
        unlock = "-rgc" in argv
        mhz = None if unlock else int(argv[argv.index("-lgc") + 1])
        self.calls.append((idx, mhz))
        if (unlock and idx in self.fail_unlock_on) or (not unlock and idx in self.fail_on):
            r.returncode, r.stderr = 1, f"pretend failure on GPU{idx}"
            return r
        self.locked[idx] = mhz
        return r


def with_smi(fake):
    ww.subprocess.run = fake.run
    return fake


def test_restore_after_exception():
    fake = with_smi(FakeSmi())
    try:
        with ww.Restore([0, 1]):
            ww.clock_lock([0, 1], 2100)
            raise ValueError("something blew up mid-sweep")
    except ValueError:
        pass
    check(fake.locked == {0: None, 1: None},
          f"after an exception the clocks must be free, got {fake.locked}")


def test_restore_after_normal_exit():
    fake = with_smi(FakeSmi())
    with ww.Restore([0, 1]):
        ww.clock_lock([0, 1], 2100)
    check(fake.locked == {0: None, 1: None}, "a clean exit must still unlock")


def test_restore_after_sigint():
    fake = with_smi(FakeSmi())
    try:
        with ww.Restore([0, 1]):
            ww.clock_lock([0, 1], 2100)
            os.kill(os.getpid(), signal.SIGINT)     # the handler raises KeyboardInterrupt
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    check(fake.locked == {0: None, 1: None}, "Ctrl-C must leave the clocks free")


def test_restore_tries_every_gpu_even_if_one_fails():
    fake = with_smi(FakeSmi(fail_unlock_on=[0]))
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        with ww.Restore([0, 1]):
            ww.clock_lock([1], 2100)
    check(fake.locked.get(1) is None,
          "GPU1 must be unlocked even though GPU0's unlock failed")
    check("nvidia-smi -rgc" in err.getvalue(),
          "when a restore fails the user must be told how to fix it by hand")


def test_second_signal_does_not_interrupt_cleanup():
    fake = with_smi(FakeSmi())
    r = ww.Restore([0, 1])
    r.__enter__()
    ww.clock_lock([0, 1], 2100)
    r._cleaning = True
    r._on_signal()            # must NOT raise: cleanup in progress
    r._cleaning = False
    with contextlib.redirect_stderr(io.StringIO()):
        r.__exit__(None, None, None)
    check(fake.locked == {0: None, 1: None}, "a second signal must not abort the restore")


def test_partial_lock_is_rolled_back():
    fake = with_smi(FakeSmi(fail_on=[1]))
    try:
        ww.clock_lock([0, 1], 2100)
        check(False, "a failing GPU must raise")
    except RuntimeError:
        pass
    check(fake.locked.get(0) is None,
          f"GPU0 was locked while GPU1 failed: it must be rolled back, got {fake.locked}")


for t in (test_restore_after_exception, test_restore_after_normal_exit,
          test_restore_after_sigint, test_restore_tries_every_gpu_even_if_one_fails,
          test_second_signal_does_not_interrupt_cleanup, test_partial_lock_is_rolled_back):
    t()


# ══════════════════════════ 2. safety: the temperature limit ════════════════

check(ww.too_hot([{"i": 0, "temp": ww.TEMP_ABORT}]) == 0, "the abort temperature must abort")
check(ww.too_hot([{"i": 0, "temp": ww.TEMP_ABORT - 1}]) is None, "one degree below must not")
check(ww.too_hot([{"i": 0, "temp": None}]) is None, "a missing temperature is not an abort")
check(ww.too_hot([{"i": 0, "temp": 50}, {"i": 1, "temp": 99}]) == 1,
      "the hot card must be found wherever it is in the list")


def test_watch_aborts_during_settle():
    """The settle happens right after a clock change - the likeliest moment for a
    thermal overshoot - and used to be completely unwatched."""
    ww.sample = lambda idx: [{"i": 0, "temp": 90, "watt": 100, "clock": 2000, "fan": 50}]
    try:
        ww.watch([0], 5)
        check(False, "watch() must abort on an overheating card")
    except RuntimeError as e:
        check("ABORT" in str(e), f"the abort must say so: {e}")


def test_measure_point_aborts_during_window():
    ww.sample = lambda idx: [{"i": 0, "temp": 95, "watt": 100, "clock": 2000, "fan": 50}]
    with_smi(FakeSmi())
    try:
        ww.measure_point([0], DummyLoad(), 2100, 0, 10)
        check(False, "measure_point must abort on an overheating card")
    except RuntimeError as e:
        check("ABORT" in str(e), f"the abort must say so: {e}")


class DummyLoad:
    running = True

    def check(self):
        pass

    def mean_between(self, a, b):
        return 100.0, 3

    def stop(self, timeout=0):
        pass

    def start(self):
        pass


test_watch_aborts_during_settle()
test_measure_point_aborts_during_window()


# ═══════════════════════ 3. a missing reading is never a zero ═══════════════

check(ww._num("[N/A]") is None, "[N/A] must be None, never 0.0")
check(ww._num("Unknown Error") is None, "an unparsable field must be None")
check(ww._num("250.5") == 250.5, "a real number must survive")


def test_point_refuses_to_guess_missing_power():
    """A card reporting [N/A] watts used to become 0 W: a machine that looks
    wonderfully efficient, and an `eco` profile pointing at the wrong clock."""
    ww.sample = lambda idx: [{"i": 0, "temp": 60, "watt": 200.0, "clock": 2000, "fan": 40},
                             {"i": 1, "temp": 60, "watt": None, "clock": 2000, "fan": 40}]
    with_smi(FakeSmi())
    try:
        ww.measure_point([0, 1], DummyLoad(), 2100, 0, 5)
        check(False, "a point without power from one GPU must be refused, not invented")
    except RuntimeError as e:
        check("power reading" in str(e), f"the reason must be explicit: {e}")


test_point_refuses_to_guess_missing_power()
ww.sample = ww.sample          # (left as the stub; no further telemetry tests below)


# ══════════════════════════ 4. the load generator ═══════════════════════════

def test_stop_waits_for_the_request_in_flight():
    """Without the join, the old thread keeps generating into the next phase:
    two concurrent loads, and a measurement of something that never happened."""
    seen = []
    L = ww.Load("http://x", "m")
    L.request = lambda prompt, tokens: (seen.append(threading.current_thread().name),
                                        time.sleep(0.3),
                                        {"_started": time.time(), "_elapsed": 0.3,
                                         "usage": {"completion_tokens": 10}})[-1]
    L.start()
    time.sleep(0.1)
    L.stop()
    check(not L.running, "after stop() no load thread may be alive")
    L.start()
    time.sleep(0.5)
    L.stop()
    check(len(set(seen)) == 1, f"two load threads ran at once: {sorted(set(seen))}")


def test_start_refuses_a_second_thread():
    L = ww.Load("http://x", "m")
    L.request = lambda p, t: (time.sleep(0.2),
                              {"_started": time.time(), "_elapsed": 0.2,
                               "usage": {"completion_tokens": 10}})[-1]
    L.start()
    try:
        L.start()
        check(False, "start() must refuse while a thread is running")
    except RuntimeError:
        pass
    L.stop()


def test_straddling_request_is_not_counted():
    """A request spanning a clock change belongs to neither point."""
    L = ww.Load("http://x", "m")
    L.samples = [(10.0, 20.0, 100.0),    # entirely inside
                 (19.0, 31.0, 50.0),     # finishes after the window: straddles
                 (5.0, 12.0, 10.0)]      # started before the window
    mean, n = L.mean_between(9.0, 30.0)
    check((mean, n) == (100.0, 1), f"only the contained request counts, got {mean} over {n}")


def test_transient_error_does_not_poison_later_points():
    L = ww.Load("http://x", "m")
    L._error = "URLError: pretend outage"
    try:
        L.check()
        check(False, "a live error must be raised")
    except RuntimeError:
        pass
    calls = {"n": 0}

    def one(prompt, tokens):
        calls["n"] += 1
        return {"_started": time.time(), "_elapsed": 0.05, "usage": {"completion_tokens": 10}}

    L.request = one
    L.start()
    time.sleep(0.3)
    L.stop()
    L.check()          # must no longer raise: a good answer cleared the error
    check(calls["n"] > 0, "the load thread must have run")


def test_prefill_is_none_not_zero_when_unavailable():
    """The reading column is advertised for vLLM/TGI/Ollama too. Reporting 0
    there reads as 'infinitely slow at reading', which nobody can spot."""
    check(ww.Load.prefill_tps({"_elapsed": 2.0, "usage": {"completion_tokens": 8}}) is None,
          "without prompt_tokens the reading speed is unknown, not zero")
    got = ww.Load.prefill_tps({"_elapsed": 2.0, "usage": {"prompt_tokens": 1000}})
    check(got == 500.0, f"reading speed must be computed when the token count is there: {got}")
    check(ww.Load.prefill_tps({"_elapsed": 1.0, "timings": {"prompt_per_second": 42.0}}) == 42.0,
          "llama.cpp's own number must win")
    check(ww.Load.prefill_tps({"_elapsed": 1.0, "usage": {"prompt_tokens": None}}) is None,
          "a null token count must not crash or become zero")


def test_decode_tps_survives_a_null_token_count():
    check(ww.Load.decode_tps({"_elapsed": 2.0, "usage": {"completion_tokens": None}}, 400)
          == 200.0, "a null completion_tokens must fall back to what we asked for")


for t in (test_stop_waits_for_the_request_in_flight, test_start_refuses_a_second_thread,
          test_straddling_request_is_not_counted, test_transient_error_does_not_poison_later_points,
          test_prefill_is_none_not_zero_when_unavailable, test_decode_tps_survives_a_null_token_count):
    t()


# ═══════════════════════════ 5. the profile rules ═══════════════════════════

p = ww.derive(REAL)
check(p["power"]["clock"] is None, "power must be the unlocked run")
check(p["free"]["clock"] == 2700, f"free should be 2700, got {p['free']['clock']}")
check(p["quiet"]["clock"] == 2400, f"quiet should be 2400, got {p['quiet']['clock']}")
check(p["eco"]["clock"] == 2100, f"eco should be 2100, got {p['eco']['clock']}")

# The rounding rule, tested with the numbers that actually need it.
check(round(117.17 / 123.34 * 100) == 95, "this point prints as 95%")
check(117.17 / 123.34 < 0.95, "...while a naive ratio comparison rejects it")

flat = {"points": [pt(None, 100, 500), pt(2400, 100, 400), pt(1500, 100, 300)]}
p = ww.derive(flat)
check(p["free"]["clock"] == 1500, "on a flat curve free must take the lowest clock")
check(p["quiet"]["clock"] == 1500, "quiet lands there too: both names stay valid")
check(p["eco"]["clock"] == 1500, "and so does eco")

prop = {"points": [pt(None, 120, 480), pt(2400, 96, 384), pt(1800, 72, 288)]}
p = ww.derive(prop)
check("free" not in p, "on a proportional curve there is no free point")
check("quiet" not in p, "nor a point costing 5% or less")
check(p["eco"]["clock"] is None, "with equal efficiency everywhere eco is the free run")

eff = {"points": [pt(None, 100, 500), pt(2400, 95, 300), pt(1500, 60, 290)]}
p = ww.derive(eff)
check(p["eco"]["clock"] == 2400, "eco took the lowest clock instead of the most efficient")
check(p["quiet"]["clock"] == 2400, "quiet lands on the same point and stays addressable")

# No unlocked run in the file: `power` must not be invented, and `set power`
# must still mean "no lock" rather than whatever the table happened to show.
headless = {"points": [pt(2700, 123.4, 465), pt(2400, 117.2, 367), pt(2100, 105.1, 303)]}
p = ww.derive(headless)
check("power" not in p, "without an unlocked run there is no power profile")
check(ww.resolve(headless, "power") is None, "`power` always means: no lock")
out = io.StringIO()
ww.load_data = lambda _: headless
with contextlib.redirect_stdout(out):
    ww.cmd_profiles(type("A", (), {"data": None})())
check("no unlocked run" in out.getvalue(), "the table must say the baseline is not a real one")

check(ww.derive({"points": []}) == {}, "no points, no profiles")
check(ww.derive({"points": [pt(None, 0, 0)]}) == {}, "a point without tokens/s is unusable")
check(list(ww.derive({"points": [pt(None, 100, 500)]})) == ["power", "eco"],
      "with only the free run, power and eco are the same single point")

check(ww.resolve(REAL, "power") is None, "power means no lock")
check(ww.resolve({}, "2200") == 2200, "a raw MHz value works without measurements")
check(ww.resolve(REAL, "eco") == 2100, "eco must resolve to the measured clock")
check(not ww.needs_data("power") and not ww.needs_data("2200"),
      "power and a raw MHz value must not require a measurement file")
check(ww.needs_data("eco"), "a derived profile does need the file")
try:
    ww.resolve(REAL, "turbo")
    check(False, "an unknown profile should have raised")
except SystemExit:
    pass


# ══════════════════════ 6. hostile measurement files ════════════════════════

for bad, why in (({"points": None}, "points: null"),
                 ({"points": [None]}, "points: [null]"),
                 ({"points": [{"tps": 100}]}, "a point without watt_total"),
                 ({"points": [{"tps": None, "watt_total": 1}]}, "tps: null"),
                 ({"points": [{"tps": 100, "watt_total": 0}]}, "watt_total: 0"),
                 ({"points": [{"tps": 100, "watt_total": 10, "clock": "fast"}]}, "clock: string"),
                 ({}, "an empty object"),
                 ({"points": "nope"}, "points as a string")):
    try:
        ww.derive(bad)
    except Exception as e:                                        # noqa: BLE001
        check(False, f"{why} crashed derive(): {type(e).__name__}: {e}")

ok_but_thin = {"points": [pt(None, 100, 500), {"tps": 90, "watt_total": 400, "clock": 2000}],
               "gpu": None, "when": None, "model": None}
out = io.StringIO()
ww.load_data = lambda _: ok_but_thin
try:
    with contextlib.redirect_stdout(out):
        ww.cmd_profiles(type("A", (), {"data": None})())
    check("n/a" in out.getvalue() or "-" in out.getvalue(),
          "missing optional fields must print as unknown, not crash")
except Exception as e:                                            # noqa: BLE001
    check(False, f"a thin but valid file crashed the table: {type(e).__name__}: {e}")

ww.load_data = LOAD_DATA       # back to the real reader for the file tests below
with tempfile.TemporaryDirectory() as d:
    broken = os.path.join(d, "broken.json")
    with open(broken, "w") as f:
        f.write("{not valid json")
    try:
        ww.load_data(broken)
        check(False, "invalid JSON must be refused")
    except SystemExit as e:
        check("not valid JSON" in str(e), f"the message must be readable: {e}")
    listy = os.path.join(d, "list.json")
    with open(listy, "w") as f:
        json.dump([1, 2, 3], f)
    try:
        ww.load_data(listy)
        check(False, "a JSON list is not a measurement file")
    except SystemExit:
        pass


# ═════════════════════════════ 7. the boot unit ═════════════════════════════

ww.gpu_indexes = lambda: [0, 1]
ww.shutil.which = lambda _: "/usr/bin/nvidia-smi"
ww.load_data = lambda _: REAL
out = io.StringIO()
with contextlib.redirect_stdout(out):
    ww.cmd_install(type("A", (), {"profile": "eco", "data": None, "show": True})())
unit = out.getvalue()
check(unit.count("ExecStart=") == 2, "one ExecStart per GPU")
check("-lgc 2100" in unit, "the unit must lock at eco's clock")
check(unit.count("ExecStop=") == 2 and "-rgc" in unit, "the unit must be able to restore")
check("RemainAfterExit=yes" in unit, "without RemainAfterExit systemd calls it failed")

# `install power` must REMOVE an existing unit: otherwise the old profile quietly
# comes back at the next reboot and the command looks like it did nothing.
with tempfile.TemporaryDirectory() as d:
    ww.UNIT_PATH = os.path.join(d, "wattwright.service")
    with open(ww.UNIT_PATH, "w") as f:
        f.write("[Service]\n")
    ran: list[tuple] = []
    ww.systemctl = lambda *a: (ran.append(a), True)[1]
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        rc = ww.cmd_install(type("A", (), {"profile": "power", "data": None, "show": False})())
    check(rc == 0 and not os.path.exists(ww.UNIT_PATH),
          "install power must remove the unit file")
    check(("disable", "--now", "wattwright") in ran, "it must also disable the unit")

    # ...and with no unit installed it must simply say so, not fail.
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        rc = ww.cmd_install(type("A", (), {"profile": "power", "data": None, "show": False})())
    check(rc == 0 and "nothing to install" in out.getvalue(),
          "install power with no unit present must be a clean no-op")

# A systemctl failure must not be reported as success.
ww.UNIT_PATH = os.path.join(tempfile.mkdtemp(), "wattwright.service")
ww.systemctl = lambda *a: False
out, err = io.StringIO(), io.StringIO()
with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
    rc = ww.cmd_install(type("A", (), {"profile": "eco", "data": None, "show": False})())
check(rc == 1, f"a systemctl failure must give a non-zero exit code, got {rc}")
check("systemd did not accept" in err.getvalue(), "and must say what went wrong")

print("FAILED:", failed) if failed else print("all ok")
sys.exit(1 if failed else 0)
