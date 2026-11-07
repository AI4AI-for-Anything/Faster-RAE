"""Per-step torch_xla counter deltas and Python GC time, shared by the step scripts.

Every counter that moved during a pipelined window is reported per step, so a
CPU fallback (`aten::*`), a host fetch (TransferFromDevice) or an extra
execution inside the step is visible by name instead of guessed. GC time comes
from gc.callbacks, which fire around every collection.
"""
import gc
import time

import torch_xla.debug.metrics as tmetrics


class Probe:
    def __init__(self):
        self.gc_ms = 0.0
        self._t = None
        gc.callbacks.append(self._cb)

    def _cb(self, phase, info):
        if phase == "start":
            self._t = time.perf_counter()
        elif self._t is not None:
            self.gc_ms += 1000 * (time.perf_counter() - self._t)
            self._t = None

    def start(self):
        self.gc_ms = 0.0
        self.snap = {k: tmetrics.counter_value(k) for k in tmetrics.counter_names()}

    def report(self, steps):
        moved = {k: round((tmetrics.counter_value(k) - self.snap.get(k, 0)) / steps, 2)
                 for k in tmetrics.counter_names()
                 if tmetrics.counter_value(k) != self.snap.get(k, 0)}
        keep = {k: v for k, v in moved.items()
                if k.startswith("aten::") or k in ("ExecuteComputation", "ExecuteReplicated",
                                                   "TransferFromDevice", "TransferToDevice",
                                                   "UncachedCompile", "CachedCompile")}
        return dict(counters_per_step=keep, gc_ms_per_step=round(self.gc_ms / steps, 2))
