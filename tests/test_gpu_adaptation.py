"""CPU checks for phase isolation and the monitored launcher's failure boundaries."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('adapt_gpu', ROOT/'tools/adapt_gpu.py')
adapt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapt)


class ConfigTests(unittest.TestCase):
    def test_phase_config_preserves_source_and_model(self):
        from omegaconf import OmegaConf
        original = OmegaConf.load(ROOT/'configs/stage1/training/DINOv2-B_decXL.yaml')
        before = OmegaConf.to_container(original, resolve=True)
        for phase, expected in [('recon', (9, 9)), ('disc', (0, 9)), ('gan', (0, 0))]:
            result = adapt.make_config(original, 32, phase, 8, 0, '/tmp/disc.pth')
            self.assertEqual(result.training.global_batch_size, 512)
            self.assertEqual(result.training.micro_batch_size, 32)
            self.assertEqual(result.training.gradient_accumulation_steps, 2)
            self.assertEqual(result.training.expected_world_size, 8)
            self.assertEqual(result.training.checkpoint_interval, 0)
            self.assertEqual(result.training.sample_every, 0)
            self.assertEqual((result.gan.loss.disc_upd_start, result.gan.loss.disc_start), expected)
            self.assertEqual(OmegaConf.to_container(result.stage_1), before['stage_1'])
            self.assertEqual(result.gan.loss.disc_weight, before['gan']['loss']['disc_weight'])
            self.assertEqual(result.training.optimizer.lr, before['training']['optimizer']['lr'])
        self.assertEqual(OmegaConf.to_container(original, resolve=True), before)

    def test_budget_overrun_terminates_owned_process(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(adapt, 'gpu_state', return_value=('fake', 140, 81)):
                result = adapt.run_monitored([sys.executable, '-c', 'import time; time.sleep(30)'],
                                             {}, Path(directory)/'log', 'fake', 80, 60)
            self.assertEqual(result['monitor_reason'], 'DEVICE_BUDGET_EXCEEDED')
            self.assertNotEqual(result['returncode'], 0)
            self.assertLess(result['wall_seconds'], 15)

    def test_timeout_is_not_success_or_oom(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(adapt, 'gpu_state', return_value=('fake', 140, 0)):
                result = adapt.run_monitored([sys.executable, '-c', 'import time; time.sleep(30)'],
                                             {}, Path(directory)/'log', 'fake', 80, .05)
            self.assertEqual(result['monitor_reason'], 'TIMEOUT')
            self.assertNotEqual(result['returncode'], 0)


if __name__ == '__main__':
    unittest.main()
