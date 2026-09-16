# wattwright

Measure **your** machine's clock / watt / token curve, then run it at the point you actually want.

GPUs ship tuned for benchmark scores, not for a box that generates tokens all day in a room
where somebody works. Capping the clock usually buys a lot of watts and quiet for very little
throughput — but *how much* is a property of your card, your cooling and your workload. It does
not transfer between machines, so wattwright does not ship a recommended clock. It ships the
measurement, and then applies whatever your own numbers say.

```
sudo wattwright measure --endpoint http://127.0.0.1:8080 --model my-model
wattwright profiles
sudo wattwright set quiet
sudo wattwright install eco        # make it the default at every boot
```

One file, no dependencies beyond the standard library. What you need:

- **Python 3.8+** and **`nvidia-smi`** (any NVIDIA driver that supports `-lgc`, i.e. most of the
  last decade of cards).
- **An OpenAI-compatible inference server already running, with your model loaded** — llama.cpp,
  vLLM, TGI, Ollama's compat endpoint. wattwright does not start one for you: it measures the
  machine under the workload you actually run.
- **root** for `measure`, `set` and `install`, because changing clocks does. `profiles` does not,
  and does not even need a GPU.
- `install` writes a **systemd** unit; on a machine without systemd, use `set` from your own
  startup script instead.

**The clocks it tries are derived from your card**, not from a list that happened to suit the
machine this was written on: by default 90/80/70/60/50% of your maximum SM clock, rounded to
50 MHz. On a 3090 MHz card that is 2800…1550; on a 1410 MHz one, 1250…700. Override with
`--clock` if you want specific points — anything above what the card can do is reported and
skipped rather than measured as a duplicate of the unlocked run.

---

## What it produces

Real output from a 2× RTX 5070 Ti box running Qwen3.8-27B on llama.cpp, tensor-split:

```
profile                   clock          tok/s   reading         watts     C    fan
power                      free    123.3  100%     2007      505  100%   74   67%
free                       2700    123.3  100%     1973      465   92%   72   63%
quiet                      2400    117.2   95%     1838      367   73%   68   52%
eco                        2100    105.1   85%     1655      303   60%   66   44%

rules: free = slowest clock keeping 99% of tokens/s - quiet = 95% - eco = highest tokens per watt
```

Read that table and the decision makes itself: on this machine **2700 MHz is free** — 8% fewer
watts for 0.1% fewer tokens, there is no reason not to. Going to 2400 buys real quiet for 5%.
Going to 2100 halves the fan and cuts 40% of the power for 15%.

That is *this* box. Yours will differ — that is the entire point.

## The rules

Profiles are derived from your measurements by rules that are stated up front, not chosen after
looking at the numbers:

| profile | rule |
|---|---|
| `power` | no clock lock at all |
| `free`  | the **lowest** clock that still keeps **99%** of the tokens/s |
| `quiet` | the **lowest** clock that still keeps **95%** |
| `eco`   | the **highest tokens per watt** |

Two rules can land on the same clock — that is information, not a collision. Both names stay
usable and the table groups them on one row. On a compute-bound curve, where throughput falls in
proportion to the clock, `free` and `quiet` simply **do not exist** and the tool says so instead
of inventing a recommendation.

Thresholds are compared on the same **rounded** percentage the table prints. A point displayed as
`95%` qualifies for a 95% rule. Without that, a real measurement of 94.997% gets rejected for a
thousandth while the screen says it passed.

## What it measures, and why that way

- **Generation and reading separately.** Prompt processing is compute-bound and degrades *more*
  than decoding at the same clock — on the box above, −17% against −15% at 2100 MHz. A tool that
  only reports tokens/s hides the cost on long documents, which is where you actually wait.
- **At equilibrium, never during the ramp.** Each point applies the clock, waits (90 s by
  default), and only then samples for 30 s. A GPU read while it is still settling reports numbers
  that do not exist.
- **Load running during the sample.** Telemetry is read *while* the endpoint is being hammered,
  from a separate thread — not before, not after.
- **Watts summed across cards, everything else averaged.** The mean wattage of two cards is not a
  quantity.

## Three traps this tool is built around

They all cost real measurements before the rules above existed.

1. **Fans have inertia coming down.** Sweeping from high clock to low leaves fans spinning fast
   from the previous, hotter point: the fan column of a descending sweep is an *upper bound*, not
   an equilibrium. The same machine reported a 78% duty at 2400 MHz during a descending sweep and **63%**
   when the point was measured from settled. wattwright settles every point before sampling; if
   you write your own sweep, do the same.
2. **Cold versus warm invalidates everything.** Benchmarking an image job at free clock first and
   at a locked clock second made the *locked* run look 40% faster — the first run was paying for
   model loading. Any comparison where one arm loads something the other has cached is measuring
   the load, not the clock.
3. **A lock left behind outlives the process.** `nvidia-smi -lgc` is not scoped to your program:
   if the process dies, the lock stays on the card until reboot and nobody remembers why the
   machine got slow. wattwright restores clocks in a `finally`, on `SIGTERM` and on `SIGINT`; a
   second signal cannot interrupt the cleanup; if one card fails to apply, the ones already
   changed are rolled back rather than left half-locked; and if a restore fails you are told
   which card and how to fix it by hand.

Two more guarantees worth knowing about:

- **The temperature limit watches the whole time the clock is locked** — the settle after a clock
  change and the sampling window alike, not just the sampling. A card reaching **84 °C** ends that
  point and frees the clocks before moving on. The settle is exactly when a thermal overshoot is
  most likely, so watching only the sampling window was watching the wrong minute.
- **Telemetry that does not answer is an error, not a cool GPU.** If `nvidia-smi` fails, the reading
  is empty — and an empty reading is never "too hot", so the temperature limit would switch itself
  off precisely when the instrument broke. A GPU that stops answering now stops the run.
- **Measurements are written after every point.** A sweep is hours; a failure on the last point
  used to throw away all the earlier ones.
- **A point with no tokens/s is dropped, not recorded.** If no request fits inside the sampling
  window, there is nothing to report: writing the point anyway would make the sweep look like it
  measured something it did not.

## Commands

```
wattwright measure    # run the sweep, write wattwright.json
    --endpoint URL    OpenAI-compatible server (default http://127.0.0.1:8080)
    --model NAME      model to ask the server for
    --clock 2800 2450 ...                 clocks to try (default: from your card's maximum)
    --settle 90 --window 30 --tokens 400

wattwright profiles   # derive and print the profiles (works with no GPU present)
wattwright set NAME   # apply a profile, or a raw MHz value: `set 2550`
wattwright status     # clocks, watts, temperature, fans right now
wattwright install NAME [--show]    # systemd unit so the profile survives reboot
wattwright install power            # ...and this removes it again
```

Your inference server is itself a process on the GPU, so wattwright lists what is running and
carries on. It does not refuse: it tells you what it found, because *another* workload alongside
it is what would quietly poison every number.

`profiles` deliberately needs no driver: a measurement file can be read, shared and compared on
any machine.

## Install

```bash
curl -O https://raw.githubusercontent.com/ValerioDolci/wattwright/main/wattwright
chmod +x wattwright && sudo mv wattwright /usr/local/bin/
```

Changing clocks needs root. Reading measurements does not.

## Scope and limits

- **NVIDIA only**, through `nvidia-smi -lgc`. AMD exposes different levers.
- **The reading column needs token counts.** llama.cpp reports prompt-processing speed directly;
  for other backends it is computed from `usage.prompt_tokens` and the wall clock. If a server
  reports neither, the column says `n/a` — never `0`, which would read as "infinitely slow at
  reading" and is indistinguishable from a real measurement.
- **Power limits are a separate lever and often a dead one**: on the box above, the VBIOS floor
  was 250 W while the cards drew 248 W under load, so capping power changed nothing measurable
  (−1.2 W, −0.02% tokens/s). Clock locking did the work instead. Check your own floor with
  `nvidia-smi --query-gpu=power.min_limit --format=csv` before reaching for that knob.
- **Fan control is not here on purpose.** On the machine this was built for, a custom fan curve
  turned out to be both impractical at full power (60% duty → 81 °C) and unnecessary at reduced
  clock, because the automatic curve drops below any duty you would have forced once the card
  stops producing that heat. Lower the watts and the fans follow.
- The profile applies to **the whole machine**. If you also run image or video generation, those
  are more compute-bound than token generation and will pay more than your `tok/s` column
  suggests — measure them before making a low clock your default.

## How this is tested

```
python3 tests/test_wattwright.py    # the suite: no GPU, no nvidia-smi, no endpoint
python3 tests/mutate.py             # break each guarantee and check the suite notices
```

`mutate.py` introduces **19 deliberate defects**, one at a time, and reports three outcomes:
`caught` (the guarantee is guarded), `NOT CAUGHT` (the tests do not cover it) and `STALE` (the
patch no longer matches the source, so nothing was mutated at all). The third one matters as much
as the second: a mutation whose target text has drifted applies nothing, the suite passes for the
wrong reason, and a dead check starts looking like a live one. Current state: **20 caught, 0 not
caught, 0 stale.**

Passing all of them means those specific regressions are guarded. It does not mean the code has no
bugs, and the difference is worth keeping in mind.

It also found the one bug nothing else did. Two adversarial reviews and nineteen mutations all
missed that the reading speed was measured on a prompt the server had just cached — the tool
reported 50 tok/s where an independent measurement said 2007. No stub has a cache, so no unit test
could see it; running the thing on a real GPU took five minutes and found it immediately. Unit
tests show the parts behave as you believe. Only the hardware shows the program does its job.

Two rounds of adversarial review produced this. The first found bugs in the code. The second found
bugs in the *tests*: one that passed with the fix removed because it let the code heal itself
before looking, and a mutation that silently stopped applying. Those are the worse kind — a bug in
the code bites you eventually, while a check that only looks like one persuades you there is
nothing to see.

## Licence

MIT.
