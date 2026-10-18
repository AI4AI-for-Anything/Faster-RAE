"""Opt-in single-GPU capacity instrumentation; not a numerical-equivalence evaluator."""
import json
import math
import statistics
import time
from pathlib import Path

import torch


def graph_count():
    from torch._dynamo.utils import counters
    return counters["stats"]["unique_graphs"]


class CapacityProbe:
    def __init__(self, args, device, world_size):
        if world_size != 1:
            raise ValueError("The capacity probe currently requires exactly one GPU/rank")
        if args.probe_steps <= args.probe_warmup or args.probe_warmup < 1:
            raise ValueError("Require probe_steps > probe_warmup >= 1")
        if args.probe_output is None:
            raise ValueError("--probe-output is required")
        total = torch.cuda.get_device_properties(device).total_memory / 2**30
        self.budget = min(args.memory_budget_gib, total)
        if not 0 < args.memory_reserve_gib < self.budget:
            raise ValueError("Require 0 < memory_reserve_gib < effective memory budget")
        self.allocator_limit = self.budget - args.memory_reserve_gib
        torch.cuda.set_per_process_memory_fraction(self.allocator_limit / total, device)
        torch.cuda.reset_peak_memory_stats(device)
        self.args, self.device = args, device
        self.rows = []
        self.update_counts = {"generator": 0, "discriminator": 0}
        self.profiler = None
        self.output = Path(args.probe_output)
        self.output.parent.mkdir(parents=True, exist_ok=True)

    def bind_optimizers(self, generator, discriminator):
        for name, optimizer in (("generator", generator), ("discriminator", discriminator)):
            def count_step(opt, args, kwargs, key=name):
                self.update_counts[key] += 1
            optimizer.register_step_post_hook(count_step)

    def begin_step(self):
        self.counts_before = self.update_counts.copy()
        if self.args.probe_profile and len(self.rows) == self.args.probe_warmup:
            self.profiler = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                record_shapes=True, profile_memory=True, with_stack=False,
            )
            self.profiler.__enter__()
        torch.cuda.synchronize(self.device)
        self.graphs_before = graph_count()
        self.start = time.perf_counter()

    def end_step(self, losses, decoder, discriminator, *, train_disc, use_gan, disc_updates):
        torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - self.start
        values = {key: float(value.detach()) for key, value in losses.items()}
        if not all(math.isfinite(value) for value in values.values()):
            raise FloatingPointError("Non-finite loss in capacity probe")
        updates = {key: self.update_counts[key] - self.counts_before[key] for key in self.update_counts}
        if updates != {'generator': 1, 'discriminator': disc_updates if train_disc else 0}:
            raise RuntimeError(f'Unexpected optimizer update cadence: {updates}')
        self.rows.append({
            "step": len(self.rows), "seconds": elapsed, "losses": values,
            "generator_updates": updates["generator"], "discriminator_updates": updates["discriminator"],
            "microbatches": self.batch_plan["gradient_accumulation_steps"],
            "adversarial_generator": use_gan,
            "new_graphs": graph_count() - self.graphs_before,
        })
        if self.profiler is not None:
            self.profiler.__exit__(None, None, None)
            self.profiler.export_chrome_trace(str(self.output.with_suffix(".trace.json")))
            self.output.with_suffix(".operators.txt").write_text(
                self.profiler.key_averages(group_by_input_shape=True).table(
                    sort_by="self_cuda_time_total", row_limit=30))
            self.profiler = None
        # Incremental results make a later failure diagnosable; only the last result is complete.
        complete = len(self.rows) >= self.args.probe_steps
        finite = {}
        if complete:
            for name, model in (("decoder", decoder), ("discriminator", discriminator)):
                grads = [p.grad for p in model.parameters() if p.grad is not None]
                params = [p for p in model.parameters() if p.requires_grad]
                finite[name] = {
                    "gradient_tensors": len(grads),
                    "gradients_finite": bool(grads) and all(bool(torch.isfinite(g).all()) for g in grads),
                    "parameters_finite": all(bool(torch.isfinite(p).all()) for p in params),
                }
            required = ["decoder"] + (["discriminator"] if train_disc else [])
            if not all(finite[n]["gradients_finite"] and finite[n]["parameters_finite"] for n in required):
                raise FloatingPointError("Missing/non-finite gradients or parameters")
        timed = self.rows[self.args.probe_warmup:]
        result = {
            "complete": complete, "kind": "capacity_smoke_only", "steps": self.rows,
            "batch_plan": self.batch_plan,
            "budget_gib": self.budget, "allocator_limit_gib": self.allocator_limit,
            "peak_allocated_gib": torch.cuda.max_memory_allocated(self.device) / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved(self.device) / 2**30,
            "median_step_ms": statistics.median(r["seconds"] for r in timed) * 1000 if timed else None,
            "warmup_steps": self.args.probe_warmup,
            "compiles_after_warmup": sum(r["new_graphs"] for r in timed),
            "profile_enabled": self.args.probe_profile, "finite_checks": finite,
        }
        self.output.write_text(json.dumps(result, indent=2) + "\n")
        return complete
