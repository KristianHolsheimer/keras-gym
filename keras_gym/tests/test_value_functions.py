import numpy as np
import pytest

from tensorflow import keras
from gym.envs.toy_text.frozen_lake import FrozenLakeEnv

from keras_gym.errors import BadModelOuputShapeError
from keras_gym.value_functions.base import GenericV, GenericQ, GenericQTypeII


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
        np.testing.assert_almost_equal(self.V(s), 0.23858608)


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
        np.testing.assert_almost_equal(self.Q(s, a), 0.4352397)


class TestGenericQTypeII:
    env = FrozenLakeEnv(is_slippery=False)
    env.observation_space.seed(13)
    glorot_unif = keras.initializers.glorot_uniform(seed=13)
    model = keras.Sequential(layers=[
        keras.layers.Dense(13, kernel_initializer=glorot_unif),
        keras.layers.Dense(4, kernel_initializer=glorot_unif)])
    model.compile(loss=keras.losses.mse, optimizer=keras.optimizers.SGD())
    Q = GenericQTypeII(env, model)

    def test_output_dims(self):
        assert self.Q.output_dims == self.env.action_space.n

    def test_check_model_shape(self):
        bad_shape_model = keras.Sequential(layers=[
            keras.layers.Dense(13), keras.layers.Dense(1)])
        bad_shape_model.compile(
            loss=keras.losses.mse, optimizer=keras.optimizers.SGD())

        with pytest.raises(BadModelOuputShapeError):
            GenericQTypeII(self.env, bad_shape_model)

    def test_call(self):
        s = self.env.observation_space.sample()
        np.testing.assert_array_almost_equal(
            self.Q(s),
            [0.3343867, -0.0796784, 0.513001, 0.3949425])
