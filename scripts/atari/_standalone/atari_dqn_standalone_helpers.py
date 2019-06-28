import sys
import os
import time

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from PIL import Image


def preprocessor(s):
    assert s.shape == (210, 160, 3), "bad shape"
    w, h = 80, 105
    q = np.array(Image.fromarray(s).convert('L').resize((w, h)), dtype='uint8')
    assert q.shape == (h, w), "bad shape"
    return q


def argmin(arr, axis=None, random_state=None):
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)
    if arr.ndim == 1:
        candidates = np.arange(arr.size)             # all
        candidates = candidates[arr == np.min(arr)]  # min
        if not isinstance(random_state, np.random.RandomState):
            # treat input random_state as seed
            random_state = np.random.RandomState(random_state)
        return random_state.choice(candidates)
    else:
        return np.argmin(arr, axis=axis)


def argmax(arr, axis=None, random_state=None):
    return argmin(-arr, axis=axis, random_state=random_state)


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
            if self._d[j0] or j0 - self._i in (1, 2, 3):
                continue
            j1 = (j0 + 1) % (self.capacity + 1)
            if self._d[j1]:
                continue
            j2 = (j1 + 1) % (self.capacity + 1)
            if not self.env.action_space.contains(self._a[j2]):
                print('j2', j2, self._a[j2])
                continue
            j3 = (j2 + 1) % (self.capacity + 1)
            if not self.env.action_space.contains(self._a[j3]):
                continue
                print('j2', j2, self._a[j2])
            idx.append([j0, j1, j2, j3])
            if len(idx) == 32:
                break

        if len(idx) < 32:
            raise RuntimeError("couldn't construct valid sample")

        idx = np.array(idx).T

        return {'X': idx[:3], 'A': idx[2], 'R': idx[2], 'D': idx[2],
                'X_next': idx[-3:], 'A_next': idx[3]}

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


class ExperienceTensorBuffer:
    def __init__(
            self,
            capacity=1000000,
            batch_size=32,
            session=None):

        self.capacity = int(capacity)
        self.batch_size = int(batch_size)
        self.session = K.get_session() if session is None else session

        # internal
        self._lives = 0
        self._init_cache()
        self.session.run(self._init_op)

    def add(self, s, a, r, done, info, session=None):
        lost_lives = info['ale.lives'] < self._lives
        self._lives = info['ale.lives']
        feed_dict = {
            self._placeholders['X']: preprocessor(s),
            self._placeholders['A']: a,
            self._placeholders['R']: r,
            self._placeholders['D']: done or lost_lives,
        }
        self.session.run(self.add_op, feed_dict=feed_dict)

    @property
    def add_op(self):
        if not hasattr(self, '_add_op'):
            i = self._counters['i']
            update_vars = [
                self._vars[k][i].assign(v)
                for k, v in self._placeholders.items()
            ]
            # update counters after updating values
            with tf.control_dependencies(update_vars):
                update_counters = [
                    self._counters['n'].assign(
                        tf.minimum(self._counters['n'] + 1, self.capacity)),
                    self._counters['i'].assign(
                        tf.mod(self._counters['n'] + 1, self._counters['N'])),
                ]
            self._add_op = tf.group(update_counters)  # implicitly update vars
        return self._add_op

    @property
    def idx(self):
        if not hasattr(self, '_idx'):
            j0 = tf.random.uniform(
                maxval=self._counters['n'],
                shape=(2 * self.batch_size,),
                dtype=tf.int32, name='idx/j0')
            j1 = tf.mod(j0 + 1, self._counters['N'], name='idx/j1')
            j2 = tf.mod(j0 + 2, self._counters['N'], name='idx/j2')
            j3 = tf.mod(j0 + 3, self._counters['N'], name='idx/j3')
            masks = tf.stack([
                tf.logical_not(tf.gather(self._vars['D'], j0)),
                tf.logical_not(tf.gather(self._vars['D'], j1)),
                tf.not_equal(j0, self._counters['i']),
                tf.not_equal(j1, self._counters['i']),
                tf.not_equal(j2, self._counters['i']),
                tf.not_equal(j3, self._counters['i']),
            ], axis=0)
            mask = tf.reduce_all(masks, axis=0)
            j = tf.boolean_mask(tf.stack([j0, j1, j2, j3], axis=1), mask)
            j = tf.transpose(j[:self.batch_size], (1, 0), name='idx/j')
            self._idx = {
                'X': j[:3], 'A': j[2], 'R': j[2], 'D': j[2],
                'X_next': j[-3:], 'A_next': j[3]}
        return self._idx

    def sample(self, n=32):
        with tf.variable_scope('experience_replay_buffer/sample'):
            self._sample_op = (
                tf.stop_gradient(
                    tf.transpose(
                        tf.gather(self._vars['X'], self.idx['X']),
                        (1, 2, 3, 0)),
                    name='X'),
                tf.stop_gradient(
                    tf.gather(self._vars['A'], self.idx['A']),
                    name='A'),
                tf.stop_gradient(
                    tf.gather(self._vars['R'], self.idx['R']),
                    name='R'),
                tf.stop_gradient(
                    tf.gather(self._vars['D'], self.idx['D']),
                    name='D'),
                tf.stop_gradient(
                    tf.transpose(
                        tf.gather(self._vars['X'], self.idx['X_next']),
                        (1, 2, 3, 0)),
                    name='X_next'),
                tf.stop_gradient(
                    tf.gather(self._vars['A'], self.idx['A_next']),
                    name='A_next')
            )
        return self._sample_op

    def _init_cache(self):
        N = self.capacity + 3
        variable_shapes = {'X': (N, 105, 80), 'A': (N,), 'R': (N,), 'D': (N,)}
        placehldr_shapes = {'X': (105, 80), 'A': (), 'R': (), 'D': ()}

        with tf.variable_scope('experience_replay_buffer'):
            # tensors that hold the experience data:
            with tf.variable_scope('vars'):
                self._vars = {
                    'X': tf.get_variable(
                        initializer=tf.zeros(variable_shapes['X'], tf.uint8),
                        name='X'),
                    'A': tf.get_variable(
                        initializer=tf.zeros(variable_shapes['A'], tf.uint8),
                        name='A'),
                    'R': tf.get_variable(
                        initializer=tf.zeros(variable_shapes['R'], tf.float32),
                        name='R'),
                    'D': tf.get_variable(
                        initializer=tf.zeros(variable_shapes['D'], tf.bool),
                        name='D'),
                }
            with tf.variable_scope('counters'):
                self._counters = {
                    'i': tf.get_variable(
                        initializer=tf.constant(0, tf.int32), name='i'),
                    'n': tf.get_variable(
                        initializer=tf.constant(0, tf.int32), name='n'),
                    'N': tf.constant(N, tf.int32, name='N'),
                }
            with tf.variable_scope('placeholders'):
                self._placeholders = {
                    'X': tf.placeholder(
                        shape=placehldr_shapes['X'],
                        dtype=tf.uint8, name='X'),
                    'A': tf.placeholder(
                        shape=placehldr_shapes['A'],
                        dtype=tf.uint8, name='A'),
                    'R': tf.placeholder(
                        shape=placehldr_shapes['R'],
                        dtype=tf.float32, name='R'),
                    'D': tf.placeholder(
                        shape=placehldr_shapes['D'],
                        dtype=tf.bool, name='D'),
                }

        self._init_op = tf.variables_initializer(tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES,
            scope='experience_replay_buffer'))

    def __len__(self):
        return self._len

    def __bool__(self):
        return bool(len(self))


class MaskedLoss:
    """
    Loss function for type-II Q-function.

    This loss function projects the predictions :math:`q(s, .)` onto the
    actions for which we actually received a feedback signal.

    Parameters
    ----------
    G : 1d Tensor, dtype: float, shape: [batch_size]

        The returns that we wish to fit our value function on.

    base_loss : keras loss function, optional

        Keras loss function. Default: :func:`keras.losses.mse`.

    """
    def __init__(self, G, base_loss=keras.losses.mse):
        if K.ndim(G) == 2:
            shape = K.int_shape(G)
            assert shape[1] == 1, f"bad shape: {shape}"
            G = K.squeeze(G, axis=1)
        assert K.ndim(G) == 1, "bad shape"
        self.G = K.stop_gradient(G)
        self.base_loss = base_loss

    def __call__(self, A, Q_pred):
        """
        Compute the projected MSE.

        Parameters
        ----------
        A : 2d Tensor, dtype = int, shape = [batch_size, 1]

            This is a batch of actions that were actually taken. This argument
            of the loss function is usually reserved for ``y_true``, i.e. a
            prediction target. In this case, ``A`` doesn't act as a prediction
            target but rather as a mask. We use this mask to project our
            predicted logits down to those for which we actually received a
            feedback signal.

        Q_pred : 2d Tensor, shape = [batch_size, num_actions]

            The predicted values :math:`q(s,.)`, a.k.a. ``y_pred``.

        Returns
        -------
        loss : 0d Tensor (scalar)

            The batch loss.

        """
        # input shape of A is generally [None, None]
        A.set_shape([None, 1])     # we know that axis=1 must have size 1
        A = tf.squeeze(A, axis=1)  # A.shape = [batch_size]
        A = tf.cast(A, tf.int64)   # must be int (we'll use `A` for slicing)

        # check shapes
        assert K.ndim(A) == 1, "bad shape"
        assert K.ndim(Q_pred) == 2, "bad shape"
        assert K.ndim(self.G) == 1, "bad shape"
        assert K.int_shape(self.G)[0] == K.int_shape(Q_pred)[0], "bad shape"

        # project onto actions taken: q(s,.) --> q(s,a)
        Q_pred_projected = self.project_onto_actions(Q_pred, A)

        # the actuall loss
        return self.base_loss(self.G, Q_pred_projected)

    @staticmethod
    def project_onto_actions(Y, A):
        """
        Project tensor onto specific actions taken.

        **Note**: This only applies to discrete action spaces.

        Parameters
        ----------
        Y : 2d Tensor, shape: [batch_size, num_actions]

            The tensor to project down.

        A : 1d Tensor, shape: [batch_size]

            The batch of actions used to project.

        Returns
        -------
        Y_projected : 1d Tensor, shape: [batch_size]

            The tensor projected onto the actions taken.

        """
        # *note* Please let me know if there's a better way to do this.
        batch_size = tf.cast(K.shape(A)[0], A.dtype)
        idx = tf.range(batch_size, dtype=A.dtype)
        indices = tf.stack([idx, A], axis=1)
        Y_projected = tf.gather_nd(Y, indices)  # shape: [batch_size]
        return Y_projected


class AtariDQN:
    INPUT_SHAPE = (105, 80, 3)
    UPDATE_STRATEGIES = ('q_learning', 'double_q_learning', 'sarsa')

    def __init__(
            self, env,
            gamma=0.99,
            learning_rate=1e-5,
            update_strategy='q_learning'):

        self.env = env
        self.num_actions = self.env.action_space.n
        self.gamma = float(gamma)
        self.learning_rate = float(learning_rate)
        self.update_strategy = update_strategy

        self._init_models()

        # internal
        self._x_prev0 = None
        self._x_prev1 = None

    def epsilon_greedy(self, s, is_first_step, epsilon=0.01):
        # get model input
        x = np.expand_dims(preprocessor(s), axis=0)
        if is_first_step or self._x_prev0 is None or self._x_prev1 is None:
            self._x_prev0 = x
            self._x_prev1 = x

        if is_first_step or np.random.rand() > epsilon:
            X = np.stack([self._x_prev1, self._x_prev0, x], axis=3)
            Q = self.predict_model.predict_on_batch(X)  # [1, num_actions]
            a = argmax(Q.ravel())
        else:
            a = self.env.action_space.sample()

        # prepare for next call
        self._x_prev1 = self._x_prev0
        self._x_prev0 = x

        return a

    def update(self, X, A, R, D, X_next, A_next):
        self.train_model.train_on_batch([X, R, D, X_next, A_next], A)

    def sync_target_model(self, tau=1.0):
        sess = K.get_session()
        sess.run(self._target_model_sync_op, feed_dict={self._tau: tau})

    def _init_models(self):
        # inputs
        X = keras.Input(
            name='X', shape=self.INPUT_SHAPE, dtype=tf.uint8)
        R = keras.Input(name='R', shape=(1,), dtype=tf.float32)
        D = keras.Input(name='D', shape=(1,), dtype=tf.bool)
        X_next = keras.Input(
            name='X_next', shape=self.INPUT_SHAPE, dtype=tf.uint8)
        A_next = keras.Input(name='A_next', shape=(1,), dtype=tf.int32)

        def construct_x_p(X):
            # X.shape = (None, 105, 80, 3)
            x_prev1, x_prev0, x = tf.split(tf.cast(X, tf.float32) / 255., 3, 3)
            dx = x - x_prev0                           # "velocity"
            ddx = (x - x_prev0) - (x_prev0 - x_prev1)  # "acceleration"
            return tf.concat([x, dx, ddx], axis=3)  # shape: (None, 105, 80, 3)

        # sequential model
        def layers(variable_scope):
            def v(name):
                return '{}/{}'.format(variable_scope, name)

            return [
                keras.layers.Lambda(
                    construct_x_p, name=v('construct_x_p')),
                keras.layers.Conv2D(
                    name=v('conv1'), filters=16, kernel_size=8, strides=4,
                    activation='relu'),
                keras.layers.Conv2D(
                    name=v('conv2'), filters=32, kernel_size=4, strides=2,
                    activation='relu'),
                keras.layers.Flatten(name=v('flatten')),
                keras.layers.Dense(
                    name=v('dense1'), units=256, activation='relu'),
                keras.layers.Dense(
                    name=v('outputs'), units=self.num_actions,
                    kernel_initializer='zeros')]

        # forward pass
        def forward_pass(X, variable_scope):
            Y = X
            for layer in layers(variable_scope):
                Y = layer(Y)
            return Y

        # predict
        Q = forward_pass(X, variable_scope='primary')

        # bootstrapped target
        bootstrap = K.squeeze(1 - tf.cast(D, tf.float32), axis=1)
        Q_next = forward_pass(X_next, variable_scope='target')
        R_flat = K.squeeze(R, axis=1)
        if self.update_strategy == 'q_learning':
            Q_next_proj = K.max(Q_next, axis=1)
            G = R_flat + bootstrap * self.gamma * Q_next_proj
        elif self.update_strategy == 'double_q_learning':
            Q_next_prim = forward_pass(X_next, variable_scope='primary')
            A_next_prim = tf.argmax(Q_next_prim, axis=1)
            Q_next_proj = MaskedLoss.project_onto_actions(Q_next, A_next_prim)
            G = R_flat + bootstrap * self.gamma * Q_next_proj
        elif self.update_strategy == 'sarsa':
            Q_next_proj = MaskedLoss.project_onto_actions(Q_next, A_next)
            G = R_flat + bootstrap * self.gamma * Q_next_proj
        else:
            raise ValueError(
                "bad update_strategy; valid options are: {}"
                .format(self.UPDATE_STRATEGIES))

        # models
        self.predict_model = keras.Model(inputs=X, outputs=Q)
        self.train_model = keras.Model(
            inputs=[X, R, D, X_next, A_next], outputs=Q)
        self.train_model.compile(
            loss=MaskedLoss(G, tf.losses.huber_loss),
            optimizer=keras.optimizers.Adam(self.learning_rate))

        # op for syncing target model
        self._tau = tf.placeholder(tf.float32, shape=())
        self._target_model_sync_op = tf.group(*(
            K.update(w_targ, w_targ + self._tau * (w_prim - w_targ))
            for w_prim, w_targ in zip(
                tf.get_collection(
                    tf.GraphKeys.GLOBAL_VARIABLES, scope='primary'),
                tf.get_collection(
                    tf.GraphKeys.GLOBAL_VARIABLES, scope='target'))))


class Scheduler:
    def __init__(
            self,
            T_start=0,
            ep_start=0,
            G_start=0,
            T_max=3000000,
            experience_replay_warmup_period=50000,
            evaluation_period=10000,
            target_model_sync_period=10000):

        self.reset_T(T_start)
        self.reset_ep(ep_start)
        self.reset_G(G_start)
        self.clear_frames()
        self.T_max = int(T_max)
        self.experience_replay_warmup_period = int(
            experience_replay_warmup_period)
        self.evaluation_period = int(evaluation_period)
        self.target_model_sync_period = int(target_model_sync_period)

        # internal
        self._T_epsilon = self.T
        self._T_evaluation = self.T
        self._dT_evaluation = 0
        self._T_target_model_sync = -1
        self._T_deltatime = 0
        self._ep_deltatime = 0
        self._ep_deltaT = 0

    def incr_T(self, n=1):
        self._T_deltatime = time.time() - self._T_time
        self._T_time += self._T_deltatime
        self._T += int(n)

    def incr_ep(self, n=1):
        self._ep_deltatime = time.time() - self._ep_time
        self._ep_time += self._ep_deltatime
        self._ep_deltaT = self._T - self._ep_T
        self._ep_T += self._ep_deltaT
        self._ep += int(n)

    def incr_G(self, r=1):
        self._G += float(r)

    def add_frame(self, frame):
        self._frames.append(frame)

    def reset_T(self, T=0):
        self._T_time = time.time()
        self._T = int(T)

    def reset_ep(self, ep=0):
        self._ep_time = time.time()
        self._ep_T = self._T
        self._ep = int(ep)

    def reset_G(self, G=0):
        self._G = float(G)

    def clear_frames(self, frames=None):
        self._frames = frames or []

    @property
    def T(self):
        return self._T

    @property
    def done(self):
        return self._T > self.T_max

    @property
    def ep(self):
        return self._ep

    @property
    def G(self):
        return self._G

    @property
    def epsilon(self):
        M = 1e6
        if self.T < M:
            return 1 - 0.9 * self.T / M
        if self.T < 2 * M:
            return 0.1 - 0.09 * (self.T - M) / M
        return 0.01

    @property
    def evaluate(self):
        self._dT_evaluation += self.T - self._T_evaluation
        self._T_evaluation = self.T

        eval_ = (
            self._dT_evaluation >= self.evaluation_period
            and not self.experience_replay_warmup)  # noqa: W503
        if eval_:
            self._dT_evaluation = 0
        return eval_

    @property
    def sync_target_model(self):
        self._T_target_model_sync += 1
        sync = self._T_target_model_sync % self.target_model_sync_period == 0
        if sync:
            self._T_target_model_sync = 0
        return sync

    @property
    def experience_replay_warmup(self):
        return self.T < self.experience_replay_warmup_period

    def print(self):
        dT = self._T - self._ep_T
        if not dT:
            print("<empty>")
            return

        dt = (time.time() - self._ep_time) / dT * 1e3
        avg_r = self._G / dT
        tmpl = (
            "T = {_T}, ep = {_ep}, G = {_G}, avg(r) = {avg_r:.04f}, "
            "dt = {dt:.03f}ms, epsilon = {eps:.03f}")
        print(tmpl.format(eps=self.epsilon, avg_r=avg_r, dt=dt, **vars(self)))
        sys.stdout.flush()

    def generate_gif(self):
        """
        Taken from: https://towardsdatascience.com/tutorial-double-deep-q-learning-with-dueling-network-architectures-4c1b3fb7f756  # noqa: E501
        """
        import imageio
        from skimage.transform import resize

        for i, frame in enumerate(self._frames):
            self._frames[i] = resize(
                frame, (420, 320, 3),
                preserve_range=True, order=0).astype('uint8')
        os.makedirs('data/gifs', exist_ok=True)
        imageio.mimsave(
            'data/gifs/T{:08d}_ep{:06d}.gif'.format(self.T, self.ep),
            self._frames,
            duration=1 / 30)
