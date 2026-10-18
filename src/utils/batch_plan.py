"""Pure batch arithmetic shared by training and capacity planning."""
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class BatchPlan:
    global_batch_size: int
    world_size: int
    micro_batch_size: int
    gradient_accumulation_steps: int

    @property
    def local_batch_size(self):
        return self.micro_batch_size * self.gradient_accumulation_steps

    def to_dict(self):
        return asdict(self) | {'local_batch_size': self.local_batch_size}


def batch_plan(global_batch_size, world_size, micro_batch_size=None):
    if global_batch_size <= 0 or world_size <= 0 or global_batch_size % world_size:
        raise ValueError('Positive global_batch_size must be divisible by positive world_size')
    local = global_batch_size // world_size
    micro = local if micro_batch_size is None else micro_batch_size
    if micro <= 0 or local % micro:
        raise ValueError('micro_batch_size must divide global_batch_size / world_size exactly')
    return BatchPlan(global_batch_size, world_size, micro, local // micro)


def training_batch_plan(config, world_size, probe_target_world_size=None):
    target = world_size if probe_target_world_size is None else probe_target_world_size
    expected = config.get('expected_world_size', target)
    if expected != target:
        raise ValueError(f'Config expects {expected} ranks, got {target}; rerun adaptation for this GPU count')
    global_batch = int(config.get('global_batch_size', config.get('batch_size', 16) * target))
    plan = batch_plan(global_batch, target, config.get('micro_batch_size'))
    if config.get('gradient_accumulation_steps', plan.gradient_accumulation_steps) != plan.gradient_accumulation_steps:
        raise ValueError('global_batch_size != world_size * micro_batch_size * gradient_accumulation_steps')
    return plan
