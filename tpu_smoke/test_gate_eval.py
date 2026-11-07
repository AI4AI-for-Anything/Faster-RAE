"""CPU check of tpu_smoke/gate_eval.py on synthetic trajectories (no torch needed).

Locks: the documented input format "<tag> <arm> s<seed> TRAJ {json}"; a copy of
the baseline is judged PASS, a +0.05 shift at every step is judged INCORRECT, and
the eval endpoint reads the TRAJ "eval" list.

Run from the repository root:  python -m tpu_smoke.test_gate_eval
"""
import json
import os
import random
import subprocess
import sys
import tempfile

_GATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gate_eval.py")


def _line(arm, seed, traj, evals):
    d = dict(traj=[[v, None] for v in traj], eval=evals)
    return f"w0 {arm} s{seed} TRAJ {json.dumps(d)}\n"


def _verdicts(text, *extra):
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(text)
    try:
        out = subprocess.run([sys.executable, _GATE, f.name, *extra],
                             capture_output=True, text=True, check=True).stdout
    finally:
        os.unlink(f.name)
    return {ln.split()[0]: ln.split()[2] for ln in out.splitlines() if " seeds=" in ln}


def main():
    rng = random.Random(0)
    lines = []
    for seed in range(4):
        base = [1.0 - 0.002 * t + rng.gauss(0, 0.01) for t in range(60)]
        ev = [0.9 + rng.gauss(0, 0.01) for _ in range(5)]
        lines.append(_line("up", seed, base, ev))
        lines.append(_line("same", seed, [v + rng.gauss(0, 1e-4) for v in base],
                           [v + rng.gauss(0, 1e-4) for v in ev]))
        lines.append(_line("shift", seed, [v + 0.05 for v in base], [v + 0.05 for v in ev]))
    text = "".join(lines)
    for endpoint in ("train", "eval"):
        got = _verdicts(text, "--endpoint", endpoint)
        assert got == {"same": "PASS", "shift": "INCORRECT"}, (endpoint, got)
    print("PASS gate_eval: near-copy -> PASS, +0.05 shift -> INCORRECT (train and eval endpoints)")


if __name__ == "__main__":
    main()
