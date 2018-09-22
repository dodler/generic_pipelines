from abc import abstractmethod

from sklearn.model_selection import train_test_split

from reader.image_reader import PillowReader


# todo add testing
class GenericXYDataset(object):

    def __init__(self, transform,
                 xy_paths_provider,
                 split=True,
                 x_reader=PillowReader(), y_reader=PillowReader()):
        super().__init__()

        self.mode = 'train'
        self.x_reader = x_reader
        self.y_reader = y_reader
        X = []
        y = []

        for x_path, y_path in xy_paths_provider:
            X.append(x_path)
            y.append(y_path)

        if split:
            self.train_x, self.val_x, self.train_y, self.val_y = train_test_split(X, y)
        else:
            self.train_x = X
            self.train_y = y
            self.val_x = []
            self.val_y = []

        self.transform = transform
        self.x_cache = {}
        self.y_cache = {}

    def __len__(self):
        if self.mode == 'train':
            return len(self.train_y)
        else:
            return len(self.val_y)

    def setmode(self, mode):
        self.mode = mode

    @abstractmethod
    def read_x(self, index):
        pass

    @abstractmethod
    def read_y(self, index):
        pass

    def fetch_cache(self, index):
        key = self.mode + str(index)
        return self.x_cache[key], self.y_cache[key]

    def put_cache(self, index, X, y):
        key = self.mode + str(index)
        self.x_cache[key] = X
        self.y_cache[key] = y

    def in_cache(self, index):
        key = self.mode + str(index)
        return key in self.x_cache.keys() and key in self.y_cache.keys()

    def __getitem__(self, index):
        if self.in_cache(index):
            X, y = self.fetch_cache(index)
        else:
            X = self.read_x(index)
            y = self.read_y(index)
            self.put_cache(index, X, y)

        return self.transform(X, y, self.mode)
