import warnings

import numpy as np
import gym
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import FunctionTransformer

from ..generic import GenericQTypeI, GenericQTypeII
from ..policy import PolicyQ


class Base:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        env = gym.make('CartPole-v0')
    random = np.random.RandomState(13)
    if hasattr(env.observation_space, 'n'):
        s = random.randint(env.observation_space.n)
    else:
        s = random.rand(*env.observation_space.sample().shape)
    if hasattr(env.action_space, 'n'):
        a = random.randint(env.action_space.n)
    else:
        a = random.rand(*env.observation_space.sample().shape)
    q1 = GenericQTypeI(
        env,
        SGDRegressor(),
        FunctionTransformer(lambda x: -x, validate=False)
    )
    q2 = GenericQTypeII(
        env,
        SGDRegressor(),
        FunctionTransformer(lambda x: -x, validate=False)
    )
    X1 = np.random.rand(13, 8)
    X2 = np.random.rand(13, 4)
    p1 = PolicyQ(q1, random_seed=13)
    p2 = PolicyQ(q2, random_seed=42)


class TestGenericQTypeI(Base):

    def test_modeltype(self):
        assert hasattr(self.q1, 'MODELTYPE')
        assert self.q1.MODELTYPE == 1

    def test_preprocess(self):
        np.testing.assert_array_equal(
            self.q1.preprocess(self.s, self.a).shape, [1, 8])
        np.testing.assert_almost_equal(
            self.q1.preprocess(self.s, self.a).sum(), -2.8052713613131006)

    def test_call(self):
        np.testing.assert_equal(self.q1(self.s, self.a).ndim, 0)

    def test_batch_evaluate(self):
        np.testing.assert_array_almost_equal(
            self.q1.batch_evaluate(self.X1).shape, [13])

    def test_policy(self):
        np.testing.assert_array_equal(
            [self.p1.random() for _ in range(8)],
            [1, 1, 1, 1, 0, 1, 0, 1])
        np.testing.assert_array_equal(
            [self.p1.greedy(self.s) for _ in range(8)],
            [0, 0, 0, 0, 0, 0, 0, 1])
        np.testing.assert_array_equal(
            [self.p1.epsilon_greedy(self.s, epsilon=0.5) for _ in range(8)],
            [1, 0, 0, 0, 1, 0, 1, 1])


class TestGenericQTypeII(Base):

    def test_modeltype(self):
        assert hasattr(self.q2, 'MODELTYPE')
        assert self.q2.MODELTYPE == 2

    def test_preprocess(self):
        np.testing.assert_array_equal(self.q2.preprocess(self.s).shape, [1, 4])
        np.testing.assert_almost_equal(
            self.q2.preprocess(self.s).sum(), -2.8052713613131006)

    def test_call(self):
        np.testing.assert_array_equal(self.q2(self.s).shape, [2])

    def test_batch_evaluate(self):
        np.testing.assert_array_almost_equal(
            self.q2.batch_evaluate(self.X2).shape, [13, 2])

    def test_policy(self):
        np.testing.assert_array_equal(
            [self.p2.random() for _ in range(8)],
            [0, 1, 1, 0, 0, 1, 0, 1])
        np.testing.assert_array_equal(
            [self.p2.greedy(self.s) for _ in range(8)],
            [0, 1, 0, 0, 0, 1, 0, 0])
        np.testing.assert_array_equal(
            [self.p2.epsilon_greedy(self.s, epsilon=0.5) for _ in range(8)],
            [1, 1, 1, 1, 1, 0, 1, 1])
