"""Statistical gate over paired-seed loss trajectories.

Input: lines "<tag> <arm> s<seed> TRAJ {json with traj=[[loss, dloss|None], ...]}", i.e.
the TRAJ line printed by rae_recipe_step.py --traj, prefixed with any tag (for
example the worker name), the arm name and the seed (one line per arm x seed). Arms: a baseline ("up"), candidates,
and a sensitivity control that MUST be judged INCORRECT, or the gate is too blunt
to trust.

Endpoints: mean G loss over consecutive windows of the trajectory (default 5
windows after skipping the first 10 steps). delta_e = 2 x the baseline's std
over seeds at endpoint e, frozen before any candidate is looked at. A candidate
passes when the mean paired difference (cand - base, same seed) is within delta
at EVERY endpoint; it fails when some endpoint's mean paired difference exceeds
delta and a paired t-test gives p < 0.05; otherwise it is UNRESOLVED (the
protocol then adds seeds up to 6).

Usage: python3 tpu_smoke/gate_eval.py <file> [--base up] [--skip 10] [--windows 5] [--endpoint train|eval]
"""
import argparse
import json
import math
import statistics
import sys
from collections import defaultdict


def t_p_two_sided(diffs):
    """Paired t-test p-value (two-sided), small-n Student t via the regularized beta."""
    n = len(diffs)
    if n < 2:
        return float("nan")
    m, sd = statistics.mean(diffs), statistics.stdev(diffs)
    if sd == 0:
        return 0.0 if m != 0 else 1.0
    t = m / (sd / math.sqrt(n))
    df = n - 1
    x = df / (df + t * t)
    return _betainc(df / 2, 0.5, x)


def _betainc(a, b, x, iters=200):
    # Continued fraction (Numerical Recipes betacf) for the regularized incomplete beta.
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x)
    if x > (a + 1) / (a + b + 2):
        return 1 - _betainc(b, a, 1 - x, iters)
    c, d = 1.0, 1 - (a + b) * x / (a + 1)
    d = 1 / d if abs(d) > 1e-30 else 1e30
    h = d
    for m in range(1, iters):
        m2 = 2 * m
        aa = m * (b - m) * x / ((a + m2 - 1) * (a + m2))
        d = 1 + aa * d; d = 1 / d if abs(d) > 1e-30 else 1e30
        c = 1 + aa / c if abs(c) > 1e-30 else 1e30
        h *= d * c
        aa = -(a + m) * (a + b + m) * x / ((a + m2) * (a + m2 + 1))
        d = 1 + aa * d; d = 1 / d if abs(d) > 1e-30 else 1e30
        c = 1 + aa / c if abs(c) > 1e-30 else 1e30
        de = d * c; h *= de
        if abs(de - 1) < 1e-12:
            break
    return math.exp(lbeta) * h / a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--base", default="up")
    ap.add_argument("--skip", type=int, default=10)
    ap.add_argument("--windows", type=int, default=5)
    ap.add_argument("--endpoint", default="train", choices=["train", "eval"],
                    help="train: window means of the training loss; eval: the TRAJ 'eval' list "
                         "(fixed held-out loss, rae_recipe_step --eval-every), one endpoint per entry")
    args = ap.parse_args()
    runs = defaultdict(dict)                       # arm -> seed -> traj losses
    for line in open(args.file):
        parts = line.split(None, 3)
        if len(parts) < 4 or not parts[3].startswith("TRAJ"):
            if "FAILED" in line:
                print("RUN FAILED:", line.strip()[:200])
            continue
        arm, seed = parts[1], parts[2]
        d = json.loads(parts[3][5:])
        runs[arm][seed] = d["eval"] if args.endpoint == "eval" else [r[0] for r in d["traj"]]
    if args.base not in runs:
        sys.exit(f"no baseline arm {args.base!r}; arms: {sorted(runs)}")

    def endpoints(tr):
        if args.endpoint == "eval":
            if len(tr) != args.windows:
                sys.exit(f"eval endpoint: expected {args.windows} entries, got {len(tr)}")
            return list(tr)
        body = tr[args.skip:]
        w = len(body) // args.windows
        return [statistics.mean(body[i * w:(i + 1) * w]) for i in range(args.windows)]

    base = {s: endpoints(t) for s, t in runs[args.base].items()}
    seeds = sorted(base)
    delta = [2 * statistics.stdev([base[s][e] for s in seeds]) for e in range(args.windows)]
    print(f"seeds {seeds}; baseline endpoints mean "
          f"{[round(statistics.mean(base[s][e] for s in seeds), 5) for e in range(args.windows)]}")
    print(f"delta (2 x baseline seed std, frozen) {[round(x, 5) for x in delta]}")
    for arm in sorted(runs):
        if arm == args.base:
            continue
        common = [s for s in seeds if s in runs[arm]]
        ep = {s: endpoints(runs[arm][s]) for s in common}
        verdicts, rows = [], []
        for e in range(args.windows):
            diffs = [ep[s][e] - base[s][e] for s in common]
            m = statistics.mean(diffs)
            p = t_p_two_sided(diffs)
            within = abs(m) <= delta[e]
            verdicts.append("in" if within else ("FAIL" if p < 0.05 else "unres"))
            rows.append(f"e{e}: d={m:+.5f} p={p:.3f} {verdicts[-1]}")
        overall = ("INCORRECT" if "FAIL" in verdicts else
                   "PASS" if all(v == "in" for v in verdicts) else "UNRESOLVED (add seeds)")
        print(f"{arm:>6} seeds={len(common)}  {overall}   " + "  ".join(rows))


if __name__ == "__main__":
    main()
