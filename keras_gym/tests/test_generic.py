import warnings

import numpy as np
import gym
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import FunctionTransformer

from ..value_functions import GenericQTypeI, GenericQTypeII
from ..policies import ValuePolicy


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
    p1 = ValuePolicy(q1, random_seed=13)
    p2 = ValuePolicy(q2, random_seed=42)


class TestGenericQTypeI(Base):

    def test_modeltype(self):
        assert hasattr(self.q1, 'MODELTYPE')
        assert self.q1.MODELTYPE == 1


class TestGenericQTypeII(Base):

    def test_modeltype(self):
        assert hasattr(self.q2, 'MODELTYPE')
        assert self.q2.MODELTYPE == 2
