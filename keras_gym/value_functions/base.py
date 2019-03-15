from abc import ABC, abstractmethod

import numpy as np

from ..errors import BadModelOuputShapeError


class BaseValueFunction(ABC):
    def __init__(self, env, model, bootstrap_model=None):
        self.env = env
        self.model = model
        self.bootstrap_model = bootstrap_model
        self.bootstrap = bootstrap_model is not None
        self._check_model_dimensions()

    @abstractmethod
    def __call__(self, *args):
        pass

    @abstractmethod
    def X(self, *args):
        pass

    def batch_eval(self, X):
        """
        TODO docs

        """
        pred = self.model.predict_on_batch(X)
        if pred.ndim == 2 and pred.shape[1] == 1:
            pred = np.squeeze(pred, axis=1)
        return pred

    def update(self, X, G, X_next=None, I_next=None):
        """
        TODO docs

        """
        if self.bootstrap:
            if X_next is None or I_next is None:
                raise TypeError(
                    "This is a bootstrapping value function, because it was "
                    "given a `bootstrap_model`. This means that `X_next` and "
                    "`I_next` both required arguments of the `update` method")
            self.bootstrap_model.train_on_batch([X, X_next, I_next], G)
        else:
            self.model.train_on_batch(X, G)

    def _set_input_dims(self, return_dummy_X=False):
        s = self.env.observation_space.sample()
        try:
            X = self.X(s)
        except TypeError as e:
            if "X() missing 1 required positional argument: 'a'" == e.args[0]:
                a = self.env.action_space.sample()
                X = self.X(s, a)
            else:
                raise

        # avoid overflow in model (space.sample can return very large numbers)
        X = (X - X.min()) / (X.max() - X.min())

        # set attribute
        self.input_dims = X.shape[1]

        return X

    def _check_model_dimensions(self):
        X = self._set_input_dims(return_dummy_X=True)
        if self.output_dims > 1:
            G = np.zeros((1, self.output_dims))
        else:
            G = np.zeros(1)

        # check if weights can be reset
        weights_resettable = (
            hasattr(self.model, 'get_weights') and  # noqa: W504
            hasattr(self.model, 'set_weights'))

        if weights_resettable:
            weights = self.model.get_weights()

        self.update(X, G)
        pred = self.batch_eval(X)

        if self.output_dims > 1 and pred.shape != (1, self.output_dims):
            raise BadModelOuputShapeError((1, self.output_dims), pred.shape)

        if self.output_dims == 1 and pred.shape != (1,):
            raise BadModelOuputShapeError((1,), pred.shape)

        if weights_resettable:
            self.model.set_weights(weights)
