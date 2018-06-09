import torch
import unittest
import numpy as np


class TestMetrics(unittest.TestCase):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)

    def test_mse(self):
        from generic_utils.metrics import mean_squared_error

        pred = torch.from_numpy(np.ones((64, 1)).astype(np.float))
        gt = torch.from_numpy(np.zeros((64, 1)).astype(np.float))

        assert np.allclose(np.array([1.0]), mean_squared_error(gt, pred))
