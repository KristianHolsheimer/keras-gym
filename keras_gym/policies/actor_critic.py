from tensorflow import keras
from tensorflow.keras import backend as K
from keras_gym.utils import full_contraction

from ..value_functions.generic import GenericV


class GenericActorCritic:
    """
    This is a simple wrapper class that combines a policy (actor) with a
    value function (critic) into a sigle object.

    We don't strictly need this, as our actor-critic type algorithms can take
    the policy and value function as separate arguments without an issue. There
    are situations, however, in which it is very useful to to have the policy
    and value function packaged together.

    TODO: class signature

    """
    def __init__(self, policy, value_function):
        self.policy = policy
        self.value_function = value_function

    def update(self):
        raise NotImplementedError('update')
