import unittest

from generic_utils.visualization.visualization import VisdomValueWatcher


class TestVisdomWatcher(unittest.TestCase):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        self.watcher = VisdomValueWatcher()

    def test_watch_losses(self):
        for i in range(100):
            self.watcher.log_value('train',(i,-i))
            # some how need to check the output