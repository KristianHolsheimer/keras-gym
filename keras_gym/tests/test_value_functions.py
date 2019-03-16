import numpy as np
import pytest

from tensorflow import keras
from gym.envs.toy_text.frozen_lake import FrozenLakeEnv

from keras_gym.errors import BadModelOuputShapeError
from keras_gym.losses import QTypeIIMeanSquaredErrorLoss
from keras_gym.value_functions import GenericV, GenericQ, GenericQTypeII


class TestGenericV:
    env = FrozenLakeEnv(is_slippery=False)
    env.observation_space.seed(42)
    glorot_unif = keras.initializers.glorot_uniform(seed=42)
    model = keras.Sequential(layers=[
        keras.layers.Dense(13, kernel_initializer=glorot_unif),
        keras.layers.Dense(1, kernel_initializer=glorot_unif)])
    model.compile(loss=keras.losses.mse, optimizer=keras.optimizers.SGD())
    V = GenericV(env, model)

    def test_call(self):
        s = self.env.observation_space.sample()
        np.testing.assert_almost_equal(self.V(s), -0.26661164)


class TestGenericQ:
    env = FrozenLakeEnv(is_slippery=False)
    env.observation_space.seed(7)
    env.action_space.seed(13)
    glorot_unif = keras.initializers.glorot_uniform(seed=7)
    model = keras.Sequential(layers=[
        keras.layers.Dense(13, kernel_initializer=glorot_unif),
        keras.layers.Dense(1, kernel_initializer=glorot_unif)])
    model.compile(loss=keras.losses.mse, optimizer=keras.optimizers.SGD())
    Q = GenericQ(env, model)

    def test_call(self):
        s = self.env.observation_space.sample()
        a = self.env.action_space.sample()
        np.testing.assert_almost_equal(self.Q(s, a), -0.14895041)


class TestGenericQTypeII:
    env = FrozenLakeEnv(is_slippery=False)
    env.observation_space.seed(13)

    @staticmethod
    def create_model():
        glorot_unif = keras.initializers.glorot_uniform(seed=13)
        X = keras.Input(name='X', shape=[16])
        G = keras.Input(name='G', shape=[1])
        layer1 = keras.layers.Dense(13, kernel_initializer=glorot_unif)
        layer2 = keras.layers.Dense(4, kernel_initializer=glorot_unif)
        y = layer2(layer1(X))
        model = keras.Model(inputs=[X, G], outputs=y)
        model.compile(
            loss=QTypeIIMeanSquaredErrorLoss(G),
            optimizer=keras.optimizers.SGD())
        return model

    @staticmethod
    def create_bad_shape_model():
        glorot_unif = keras.initializers.glorot_uniform(seed=13)
        X = keras.Input(name='X', shape=[16])
        G = keras.Input(name='G', shape=[1])
        layer1 = keras.layers.Dense(13, kernel_initializer=glorot_unif)
        layer2 = keras.layers.Dense(1, kernel_initializer=glorot_unif)  # <---
        y = layer2(layer1(X))
        model = keras.Model(inputs=[X, G], outputs=y)
        model.compile(
            loss=keras.losses.mse,
            optimizer=keras.optimizers.SGD())
        return model

    def test_output_dim(self):
        Q = GenericQTypeII(self.env, self.create_model())
        assert Q.output_dim == self.env.action_space.n

    def test_check_model_shape(self):
        with pytest.raises(BadModelOuputShapeError):
            GenericQTypeII(self.env, self.create_bad_shape_model())

    def test_call(self):
        Q = GenericQTypeII(self.env, self.create_model())
        s = self.env.observation_space.sample()
        np.testing.assert_array_almost_equal(
            Q(s),
            [0.189891, -0.036312, -0.645451, 0.448175])
