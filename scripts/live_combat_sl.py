"""Search exact room-entry snapshots in an isolated Sts2Headless process."""
from __future__ import annotations

from hashlib import sha256
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Timer
from time import perf_counter

import numpy as np


def frame_guard(raw: bytes) -> str:
    # Observations are encoded without labels. Ignore process-local counters:
    # combat wins, actions taken, and cumulative missing vocabulary entries.
    return sha256(raw[:48] + raw[60:]).hexdigest().upper()


class CombatSearch:
    def __init__(self, cfg: dict, directory: Path):
        from sts2env.env import DEFAULT_DLL

        dll = Path(cfg.get("sl_headless_dll") or DEFAULT_DLL)
        self.dll = (directory / dll).resolve()
        self.seconds = min(5.0, max(0.1, float(cfg.get("sl_seconds", 5.0))))
        self.env = None
        self.user_dir = TemporaryDirectory(prefix="necropilot-shadow-")
        os.environ["STS2_USER_DIR"] = self.user_dir.name

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None
        self.user_dir.cleanup()

    def search(self, model, frame: bytes, room_save: str, room: str) -> dict:
        from sts2env.env import Sts2Env
        from sts2env.protocol import DecisionKind, TokenKind, Zone, parse
        from sts2rl.features import Ragged
        from sts2rl.greedy import merged_argmax

        def potions(obs):
            return int(np.count_nonzero(
                (obs.tok_ints[:, 0] == TokenKind.POTION) &
                (obs.tok_ints[:, 2] == Zone.ALLY)
            ))

        cold = self.env is None
        if cold:
            if not self.dll.is_file():
                return {"sl_status": "engine_missing"}
            # Forward engine diagnostics to the Python worker's stderr, which the Mod records.
            self.env = Sts2Env(dll=self.dll, engine="csharp", log=None)
        proc = self.env._proc

        def stop_engine():
            if proc.p.poll() is None:
                try:
                    proc.p.kill()
                except ProcessLookupError:
                    pass

        watchdog = Timer(self.seconds + (10.0 if cold else 2.0), stop_engine)
        watchdog.daemon = True
        watchdog.start()
        has_snapshot = False
        try:
            def request(**req):
                proc.send(req)
                return proc.recv()

            root = request(op="reset", mode="run", room_save=room_save, route_summary=True)
            if frame_guard(root) != frame_guard(frame):
                return {"sl_status": "opening_mismatch"}
            start = parse(root)
            if start.decision != DecisionKind.COMBAT or not start.in_combat:
                return {"sl_status": "not_combat"}
            request(op="snapshot", slot=0)
            has_snapshot = True
            deadline = perf_counter() + self.seconds

            start_pots = potions(start)
            best, best_key = [], None
            completed = 0
            rng_root = sha256(room_save.encode("utf-8"))
            for trial in range(1 + {"Monster": 4, "Elite": 12, "Boss": 20}[room]):
                if perf_counter() >= deadline:
                    break
                raw = root if trial == 0 else request(op="restore", slot=0)
                obs = parse(raw)
                trial_hash = rng_root.copy()
                trial_hash.update(trial.to_bytes(4, "little"))
                seed = trial_hash.digest()
                rng = np.random.default_rng(int.from_bytes(seed[:8], "little"))
                plan = []
                for _ in range(400):
                    if obs.done or obs.stuck or not obs.in_combat or obs.combats_won != start.combats_won:
                        break
                    if not obs.num_actions or perf_counter() >= deadline:
                        break
                    if obs.num_actions == 1:
                        action = 0
                    else:
                        logits, _ = model.run(Ragged.from_obs([obs], mode=np.ones(1, dtype=np.int64)))
                        scores = logits[0, :obs.num_actions].astype(np.float64)
                        if trial:
                            temp = (0.5, 0.8, 1.1)[(trial - 1) % 3]
                            weights = np.exp((scores - scores.max()) / temp)
                            action = int(rng.choice(obs.num_actions, p=weights / weights.sum()))
                        else:
                            weights = np.exp(scores - scores.max())
                            action = int(merged_argmax(obs, weights / weights.sum()))
                    plan.append({"guard": frame_guard(raw), "index": action})
                    raw = request(op="step", action=action)
                    obs = parse(raw)

                won = obs.victory if obs.done else (
                    not obs.stuck and (obs.combats_won > start.combats_won or not obs.in_combat)
                )
                if not won:
                    if obs.done or obs.stuck:
                        completed += 1
                    continue
                completed += 1
                end_pots = potions(obs)
                score = (obs.hp + 0.20 * start.max_hp * end_pots, obs.hp, -len(plan))
                if best_key is None or score > best_key:
                    best, best_key = plan, score
                if obs.victory or (obs.hp >= start.hp and end_pots >= start_pots):
                    break
            return {"sl_status": "planned" if best else "no_win", "sl_plan": best, "sl_completed": completed}
        except Exception:
            stop_engine()
            raise
        finally:
            try:
                if has_snapshot and proc.p.poll() is None:
                    try:
                        self.env.drop(0)
                    except (OSError, RuntimeError):
                        stop_engine()
            finally:
                watchdog.cancel()
                watchdog.join()
                if proc.p.poll() is not None:
                    self.env = None
