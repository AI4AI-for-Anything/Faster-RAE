"""Check batch invariants and actual gradients, including two-rank CPU DDP."""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from utils.accumulation import generator_backward, sync_context
from utils.batch_plan import batch_plan, training_batch_plan


def losses(model, data):
    prediction = model(data)
    rec = (prediction - 0.7).square().mean()
    gan = (prediction + 0.2).square().mean()
    return {'recon_total': rec, 'recon': rec, 'lpips': rec * 0, 'gan': gan}


def backward(model, chunks, use_gan=True):
    raw = getattr(model, 'module', model)
    return generator_backward(model, chunks, losses, raw[-1].weight,
                              use_gan=use_gan, disc_weight=.75, max_d_weight=1e4)


def ddp_worker(rank, rendezvous):
    dist.init_process_group('gloo', init_method='file://' + rendezvous, rank=rank, world_size=2)
    try:
        torch.manual_seed(3)
        model = torch.nn.Sequential(torch.nn.Linear(3, 5), torch.nn.Tanh(), torch.nn.Linear(5, 2)).double()
        reference = deepcopy(model)
        ddp = DDP(model)
        data = torch.arange(24, dtype=torch.float64).reshape(8, 3) / 24 + rank
        # Reference: rank-local adaptive weight, then the mean of rank gradients.
        backward(reference, [data])
        for parameter in reference.parameters():
            dist.all_reduce(parameter.grad)
            parameter.grad /= 2
        backward(ddp, data.split(2))
        for actual, expected in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-10, atol=1e-10)
        optimizer = torch.optim.SGD(ddp.parameters(), lr=.1)
        optimizer.step()
        # A second iteration catches stale DDP reducer state.
        optimizer.zero_grad(set_to_none=True)
        backward(ddp, data.split(2))
        optimizer.step()
        for parameter in model.parameters():
            other = parameter.detach().clone()
            dist.broadcast(other, src=0)
            torch.testing.assert_close(parameter, other, rtol=0, atol=0)
        # Exercise the D loop's no_sync scaling with different inputs per rank.
        optimizer.zero_grad(set_to_none=True)
        for index, batch in enumerate(data.split(2)):
            with sync_context(ddp, index == 3):
                (losses(ddp, batch)['gan'] / 4).backward()
        optimizer.step()
        for parameter in model.parameters():
            other = parameter.detach().clone()
            dist.broadcast(other, src=0)
            torch.testing.assert_close(parameter, other, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


class AccumulationTests(unittest.TestCase):
    def test_batch_plans_preserve_global(self):
        for world in (1, 2, 4, 8, 16):
            for micro in (8, 16, 32):
                plan = batch_plan(512, world, micro)
                self.assertEqual(plan.world_size * plan.micro_batch_size * plan.gradient_accumulation_steps, 512)
        self.assertEqual(batch_plan(512, 8, 32).gradient_accumulation_steps, 2)
        for args in ((512, 3, 32), (512, 8, 48), (512, 8, 128), (512, 0, 32)):
            with self.assertRaises(ValueError):
                batch_plan(*args)

    def test_launch_mismatch_fails_and_probe_is_explicit(self):
        config = dict(global_batch_size=512, expected_world_size=8,
                      micro_batch_size=32, gradient_accumulation_steps=2)
        with self.assertRaises(ValueError):
            training_batch_plan(config, 1)
        self.assertEqual(training_batch_plan(config, 1, 8).local_batch_size, 64)
        with self.assertRaises(ValueError):
            training_batch_plan(config | {'gradient_accumulation_steps': 3}, 8)

    def test_accumulated_gradients_and_adaptive_weight_match_full_local_batch(self):
        for gan in (False, True):
            torch.manual_seed(4)
            model = torch.nn.Sequential(torch.nn.Linear(3, 5), torch.nn.Tanh(), torch.nn.Linear(5, 2)).double()
            reference = deepcopy(model)
            data = torch.randn(8, 3, dtype=torch.float64)
            full_metrics, full_weight = backward(reference, [data], gan)
            metrics, weight = backward(model, data.split(2), gan)
            torch.testing.assert_close(weight, full_weight, rtol=1e-10, atol=1e-10)
            for actual, expected in zip(model.parameters(), reference.parameters()):
                torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-10, atol=1e-10)
            torch.testing.assert_close(metrics['total'], full_metrics['total'], rtol=1e-10, atol=1e-10)

    def test_rng_replay_reuses_stochastic_inputs(self):
        model = torch.nn.Sequential(torch.nn.Linear(3, 2)).double()
        chunks = torch.ones(8, 3, dtype=torch.float64).split(2)
        seen = []
        def stochastic(module, batch):
            noise = torch.randn_like(batch)
            seen.append(noise.clone())
            return losses(module, batch + noise)
        generator_backward(model, chunks, stochastic, model[-1].weight,
                           use_gan=True, disc_weight=.75, max_d_weight=1e4)
        self.assertEqual(len(seen), 8)
        for first, replay in zip(seen[:4], seen[4:]):
            torch.testing.assert_close(first, replay, rtol=0, atol=0)

    def test_two_rank_ddp_gradients_and_parameter_agreement(self):
        with tempfile.TemporaryDirectory() as folder:
            mp.spawn(ddp_worker, args=(str(Path(folder) / 'rendezvous'),), nprocs=2, join=True)


if __name__ == '__main__':
    unittest.main()
