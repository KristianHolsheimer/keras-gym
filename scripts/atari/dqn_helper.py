import numpy as np
from PIL import Image


def preprocessor(s):
    assert s.shape == (210, 160, 3), "bad shape"
    w, h = 80, 105
    q = np.array(Image.fromarray(s).convert('L').resize((w, h)), dtype='uint8')
    assert q.shape == (h, w), "bad shape"
    return q


class ExperienceArrayBuffer:
    def __init__(self, env, capacity=1000000, random_state=None):
        self.env = env
        self.capacity = int(capacity)

        # set random state
        if isinstance(random_state, np.random.RandomState):
            self.random = random_state
        else:
            self.random = np.random.RandomState(random_state)

        # internal
        self._i = 0
        self._len = 0
        self._lives = 0
        self._init_cache()

    def add(self, s, a, r, done, info):
        x = preprocessor(s)
        self._x[self._i] = x
        self._a[self._i] = a
        self._r[self._i] = r
        self._d[self._i] = done
        self._i = (self._i + 1) % (self.capacity + 1)
        if self._len < self.capacity:
            self._len += 1

    def idx(self, n=32):
        idx = []
        for attempt in range(256):
            j0 = self.random.randint(len(self))
            if self._d[j0] or j0 - self._i in (1, 2, 3, 4):
                continue
            j1 = (j0 + 1) % (self.capacity + 1)
            if self._d[j1]:
                continue
            j2 = (j1 + 1) % (self.capacity + 1)
            if not self.env.action_space.contains(self._a[j2]):
                continue
            j3 = (j2 + 1) % (self.capacity + 1)
            if not self.env.action_space.contains(self._a[j3]):
                continue
            j4 = (j3 + 1) % (self.capacity + 1)
            if not self.env.action_space.contains(self._a[j4]):
                continue
            idx.append([j0, j1, j2, j3, j4])
            if len(idx) == 32:
                break

        if len(idx) < 32:
            raise RuntimeError("couldn't construct valid sample")

        idx = np.array(idx).T

        return {'X': idx[:4], 'A': idx[3], 'R': idx[3], 'D': idx[3],
                'X_next': idx[-4:], 'A_next': idx[4]}

    def sample(self, n=32):
        idx = self.idx(n=n)
        X = self._x[idx['X']].transpose((1, 2, 3, 0))
        A = self._a[idx['A']]
        R = self._r[idx['R']]
        D = self._d[idx['D']]
        X_next = self._x[idx['X_next']].transpose((1, 2, 3, 0))
        A_next = self._a[idx['A_next']]
        return X, A, R, D, X_next, A_next

    def _init_cache(self):
        n = (self.capacity + 1,)
        shape = n + (105, 80)
        self._x = np.empty(shape, 'uint8')  # frame
        self._a = np.empty(n, 'int32')      # actions taken
        self._r = np.empty(n, 'float')      # rewards
        self._d = np.empty(n, 'bool')       # done?

    def __len__(self):
        return self._len

    def __bool__(self):
        return bool(len(self))
