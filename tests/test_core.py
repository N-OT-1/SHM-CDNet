"""CPU-only checks; full CUDA training is intentionally not started by tests."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from shm.data import LoadCDPair
from shm.metrics import confusion_counts, foreground_metrics
from shm.model import GatedDifferenceModule


class CoreTests(unittest.TestCase):
    def test_metrics(self):
        counts = confusion_counts(np.array([[1, 1, 0, 0]]), np.array([[1, 0, 1, 255]]))
        self.assertEqual(counts.tolist(), [1, 1, 1, 0])
        row = foreground_metrics(counts)
        self.assertAlmostEqual(row['f1'], 2*row['iou']/(1+row['iou']))

    def test_empty_metric_denominators(self):
        row = foreground_metrics([0, 0, 0, 5])
        self.assertEqual(row['f1'], 0)
        self.assertEqual(row['oa'], 1)

    def test_loader_preserves_pair_and_mask_alignment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = np.full((8, 8, 3), [10, 20, 30], dtype=np.uint8)
            second = np.full((8, 8, 3), [40, 50, 60], dtype=np.uint8)
            mask = np.zeros((8, 8), dtype=np.uint8); mask[:, 4:] = 255
            for name, array in [('a.png', first), ('b.png', second), ('mask.png', mask)]:
                Image.fromarray(array).save(root/name)
            loaded = LoadCDPair().transform(dict(img_path=str(root/'a.png'), img_path2=str(root/'b.png'), seg_map_path=str(root/'mask.png')))
            np.testing.assert_array_equal(loaded['img'][..., :3], first)
            np.testing.assert_array_equal(loaded['img'][..., 3:], second)
            np.testing.assert_array_equal(loaded['gt_seg_map'], mask//255)

    def test_parallel_gate_backward(self):
        module = GatedDifferenceModule(8, 8, gate_channels='channel_spatial',
                                      norm_cfg=dict(type='GN', num_groups=4))
        first = torch.randn(2, 8, 8, 8, requires_grad=True)
        second = torch.randn_like(first)
        output = module(first, second)
        self.assertEqual(tuple(output.shape), (2, 8, 8, 8))
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(first.grad).all())


if __name__ == '__main__':
    unittest.main()
