A. Verdetti

1. RISOLTO A META' — `clock_lock()` fa rollback solo su `RuntimeError` e ignora i fallimenti di `_lock_one(i, None)`; `cmd_set()` non ha `Restore`, quindi un Ctrl-C durante il lock non viene ripulito.
2. RISOLTO A META' — `Restore.__enter__` installa ancora solo SIGTERM/SIGINT; per questi, `_cleaning` protegge solo il parent, non il child `nvidia-smi` lanciato da `unlock_all()`.
3. RISOLTO — `Load.start()` rifiuta se `running`, `stop()` fa `join`; nel flusso attuale start/stop sono solo sul main thread.
4. RISOLTO A META' — `Load.loop` azzera `_error` solo dentro il thread; `measure_prefill()` può fermare il thread e fare richieste dirette senza azzerarlo, lasciando un errore stale per `check()`.
5. RISOLTO — `_num()` restituisce `None`, `measure_point()` ignora i `None` e rifiuta il punto se manca `watt`; `_fmt()` ha fallback.
6. RISOLTO — `measure_point()` chiama `watch(indexes, settle, load)` prima del sampling; `watch()` chiama `too_hot(sample(indexes))`.
7. RISOLTO A META' — `prefill_tps()` non restituisce più 0, ma quando manca `timings.prompt_per_second` usa `usage.prompt_tokens / d["_elapsed"]`, cioè tempo totale prefill+decode.
8. RISOLTO — `write_data()` è chiamato dopo ogni punto; i punti precedenti restano sul file anche se lo sweep muore più avanti.
9. RISOLTO — `resolve()` restituisce `None` per `power` sempre; `derive()` crea `power` solo se esiste un punto unlocked; la tabella segnala la baseline diversa.
10. RISOLTO A META' — `install power` rimuove la unit, ma in `cmd_install()` i ritorni di `systemctl("disable", "--now", ...)` e `systemctl("daemon-reload")` sono ignorati.
11. RISOLTO — `cmd_measure()` non rifiuta per `gpu_processes()`; stampa solo il warning.
12. RISOLTO A META' — restano crash su file JSON validi ma ostili: `usable_points()` su `points` int/bool, `cmd_profiles()` su `gpu` int, `int(clock_value)` su `clock` NaN/Inf.

B. Regressioni

- `cmd_measure()`: il warmup `watch(indexes, 60, load)` è fuori dal `try/except` dei punti. Ripro: endpoint down; il thread di carico pone `_error`, `watch()` chiama `load.check()`, `RuntimeError` non gestito, traceback. I clock tornano liberi, ma il CLI crasha.
- `measure_point()` + `mean_between()`: il cambio di clock e la fine della finestra lasciano richieste in corso. Ripro: una richiesta in flight al cambio di clock, o una richiesta che finisce dopo `end`, ha il suo consumo incluso in `reads`/`watt`, ma i suoi token sono esclusi da `tps`; la curva clock/watt/token è distorta.
- `measure_point()`: se nessuna richiesta è interamente contenuta nella finestra, `tps` è `None`, ma il punto viene comunque appeso e scritto. Ripro: richieste più lunghe di `--window`; il file contiene punti con `tps: null` che `profiles` scarta, ma lo sweep li conta e li stampa.
- `measure_prefill()`: la fase stop + richieste dirette non campiona temperatura né potenza. Ripro: GPU vicina a `TEMP_ABORT`; `load.stop()` può attendere fino a 620 s e le richieste dirette con prompt lungo possono far salire la temperatura senza abort.
- `cmd_profiles()`: `groups` raggruppa per clock e stampa `p = prof[names[0]]`. Ripro: due punti con clock 2000 e watt diversi, `free` e `eco` cadono sullo stesso clock ma su punti diversi; la riga mostra i campi del primo nome, non del secondo.
- `derive()`: `unlocked = next(...)` sceglie il primo punto a clock libero. Ripro: file con due punti `clock: null`, uno 50 tok/s e uno 100 tok/s; baseline e profilo `power` sono il punto da 50, non il più veloce.

C. Bug nuovi

- Safety: `cmd_set` senza `Restore`. Ripro: `wattwright set 2100` su 2 GPU; Ctrl-C dopo il lock di GPU0 e prima di GPU1; `main` cattura `KeyboardInterrupt` e esce, GPU0 resta locked.
- Safety: secondo segnale al process group durante `Restore.__exit__`. Ripro: Ctrl-C durante la cleanup; `unlock_all()` lancia `nvidia-smi -rgc`, il child riceve SIGINT e muore, il parent non interrompe la cleanup ma l'unlock fallisce; lock resta.
- Safety: rollback di `clock_lock()` fallito non segnalato. Ripro: GPU0 lock ok, GPU1 lock fail, rollback GPU0 fail; `cmd_set()` solleva, ma stampa comunque "clocks were put back the way they were.".
- Boot: `install power` ignora fallimenti `systemctl`. Ripro: stub `systemctl` che restituisce `False`; il file viene rimosso e il comando stampa "boot profile removed" pur avendo fallito disable/daemon-reload.
- Crash: `{"points": 1}` o `{"points": true}` in `usable_points()`; `raw or []` diventa `1`/`True`, `for p in ...` solleva `TypeError`.
- Crash: `{"gpu": 1}` in `cmd_profiles()`; `data.get("gpu") or []` diventa `1`, `for g in ...` solleva `TypeError`.
- Crash: `clock` NaN/Inf. `json.load` accetta `NaN`/`Infinity`; se un profilo finisce su quel punto, `int(clock_value)` in `cmd_profiles()` solleva `ValueError`/`OverflowError`.
- Misure: `prefill_tps()` con `usage.prompt_tokens / _elapsed` totale. Ripro: backend OpenAI, `prompt_tokens=1000`, 8 token completati, elapsed 5 s; la colonna reading mostra 200 tok/s, che non è prefill reale.
- Misure: punti con `tps: null` scritti e contati. Ripro: server lento, `--window 4`, richieste da 10 s; il punto viene salvato senza tok/s e mostrato come punto misurato.
- Safety: `sample()`/`watch()` ignorano fallimenti di `nvidia-smi`. Ripro: `nvidia-smi` in errore durante settle; `sample()` restituisce `[]`, `too_hot([])` restituisce `None`, temperatura non controllata.
- Residual: `_error` stale dopo `measure_prefill()`. Ripro: errore transitorio nel thread di carico, `stop()` durante il sleep di 2 s, richiesta diretta prefill ok, thread riavviato; al punto successivo `watch()` chiama `load.check()` prima che una nuova richiesta buona azzeri l'errore e il punto viene scartato.
