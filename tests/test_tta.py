import unittest
import numpy as np

from inference import TTA


class TestTTA(unittest.TestCase):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        self.tta = TTA(lambda x: x)

    def test_tta(self):
        print('testing')
        assert (4,224,224,3) == self.tta(np.random.uniform(0, 1, (224, 224, 3)).astype(np.float)).shape
