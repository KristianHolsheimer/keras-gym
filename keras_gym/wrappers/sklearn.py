import logging

import numpy as np
from sklearn.exceptions import NotFittedError

logger = logging.getLogger()


class SklearnModelWrapper:
    """
    A simple wrapper that allows us to use a sklearn function approximator
    instead of a Keras model. This wrapper only supports the functionality
    required by the `keras-gym` package.

    Parameters
    ----------
    estimator : sklearn estimator

        This estimator must have a :term:`partial_fit` method.

    transformer : sklearn transformer, optional
        Unfortunately, there's no support for out-of-core fitting of
        transformers in scikit-learn. We can, however, use stateless
        transformers such as :py:class:`FunctionTransformer
        <sklearn.preprocessing.FunctionTransformer>`. We can also use other
        transformers that only learn the input shape at training time, such as
        :py:class:`PolynomialFeatures
        <sklearn.preprocessing.PolynomialFeatures>`. Note that these do require
        us to set `attempt_fit_transformer=True`.

    label_encoder : sklearn label_encoder, optional
        Similar to transformer, except that it's applied to the target label
        `y` instead of the input features `X`.

    attempt_fit_transformer : bool, optional
        Whether to attempt to pre-fit the transformer. Note: this is done on
        only one data point. This works for transformers that only require the
        input shape and/or dtype for fitting. In other words, this will *not*
        work for more sophisticated transformers that require batch
        aggregations.

    attempt_fit_label_encoder : bool, optional
        Whether to attempt to pre-fit the label encoder. Note: this is done on
        only one data point. This works for label encoders that only require
        the input shape and/or dtype for fitting. In other words, this will
        *not* work for more sophisticated transformers that require batch
        aggregations.

    """
    def __init__(self,
                 estimator,
                 transformer=None,
                 label_encoder=None,
                 attempt_fit_transformer=False,
                 attempt_fit_label_encoder=False):
        self.estimator = estimator
        self.transformer = transformer
        self.label_encoder = label_encoder
        self.attempt_fit_transformer = attempt_fit_transformer
        self.attempt_fit_label_encoder = attempt_fit_label_encoder

    def train_on_batch(self, X, Y, sample_weight=None):
        if Y.ndim == 2 and Y.shape[1] == 1:
            Y = np.ravel(Y)  # turn into 1d array
        X = self._transform(X)
        Y = self._encode_label(Y)
        self.estimator.partial_fit(X, Y, sample_weight=sample_weight)

    def predict_on_batch(self, X):
        X = self._transform(X)
        if hasattr(self.estimator, 'predict_proba'):
            pred = self.estimator.predict_proba(X)
        else:
            pred = self.estimator.predict(X)
        if pred.ndim == 1:
            pred = np.expand_dims(pred, axis=1)
        return pred

    def _transform(self, X):
        if self.transformer is not None:
            try:
                X = self.transformer.transform(X)
            except NotFittedError:
                if not self.attempt_fit_transformer:
                    raise NotFittedError(
                        "transformer needs to be fitted; setting "
                        "attempt_fit_transformer=True will fit the "
                        "transformer on one data point")
                logger.info(
                    "attempting to fit transformer on one single batch (this "
                    "may consist of only a single data point)")
                X = self.transformer.fit_transform(X)
        return X

    def _encode_label(self, y):
        # TODO: this is untested, must actually try it out.
        if self.label_encoder is not None:
            try:
                y = self.label_encoder.transform(y)
            except NotFittedError:
                if not self.attempt_fit_label_encoder:
                    raise NotFittedError(
                        "label_encoder needs to be fitted; setting "
                        "attempt_fit_transformer=True will fit the "
                        "label encoder on one data point")
                logger.info(
                    "attempting to fit label encoder on one single batch "
                    "(this may consist of only a single data point)")
                y = self.label_encoder.fit_transform(y)
        return y
