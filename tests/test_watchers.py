import unittest

import numpy as np
import torch

from generic_utils.output_watchers import DisplayImage


class TestWatcher(unittest.TestCase):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        self.test_input = torch.from_numpy(np.random.normal(scale=255, size=(4, 3, 384, 384)))
        self.test_target = torch.from_numpy(np.random.normal(scale=255, size=(4, 1, 384, 384)))
        self.test_output = torch.from_numpy(np.random.normal(size=(4, 1, 384, 384)))

    def test_display_image(self):
        watcher = DisplayImage('test', display_amount=2)
        watcher(self.test_input, self.test_output)

    def test_little_batch(self):
        watcher = DisplayImage('test', display_amount=8)
        watcher(self.test_input, self.test_target, self.test_output)
        watcher(self.test_input, self.test_target, self.test_output)
        watcher(self.test_input, self.test_target, self.test_output)
        watcher.close_windows()
