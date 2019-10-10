import tensorflow as tf

from .base import BaseProbaDist


class CategoricalDist(BaseProbaDist):
    """
    TODO: write docstring

    """
    def __init__(self, env, Z, with_logits=True, random_seed=None):
        self.random_seed = random_seed  # also sets self.random (RandomState)
        self.with_logits = bool(with_logits)
        if self.with_logits:
            self.Z = Z
            self.P = tf.math.softmax(Z)
            self.log_P = tf.math.log_softmax(Z)
        else:
            self.Z = None
            self.P = Z
            self.log_P = tf.math.log(Z)

    def sample(self, n=1):
        pass

    def log_proba(self, x):
        pass

    def entropy(self):
        pass

    def cross_entropy(self, other):
        pass

    def kl_divergence(self, other):
        pass

    def proba_ratio(self, other, x):
        pass

    def _check_model(self, model):
        if not isinstance(model, tf.keras.Model):
            raise TypeError(
                f"expected a keras.Model, got: {model.__class__.__name__}")
        if len(model.outputs) != 2:
            raise ValueError(
                "expected a model with two outputs (mu, logvar), the "
                f"provided model has {len(model.outputs)} outputs instead")
        if not model.output_names[0].endswith('/mu'):
            raise ValueError(r"the first output must have name {scope}/mu")
        if not model.output_names[1].endswith('/logvar'):
            raise ValueError(r"the first output must have name {scope}/logvar")
        mu, logvar = model.outputs
        check_tensor(mu, ndim=2, axis_size=self.ndim, axis=1)
        check_tensor(logvar, same_as=mu)
        self.mu = mu
        self.logvar = logvar
        self.model = model
