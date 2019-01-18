from .generic import GenericV, GenericQ
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import FunctionTransformer


class LinearV(GenericV):
    """
    A linear function approximator for a state value function :math:`V(s)`
    based on :py:class:`SGDRegressor<sklearn.linear_model.SGDRegressor>`.

    Parameters
    ----------
    env : gym environment spec
        This is used to get information about the shape of the observation
        space and action space.

    transformer : sklearn transformer, optional
        Unfortunately, there's no support for out-of-core fitting of
        transformers in scikit-learn. We can, however, use stateless
        transformers such as
        :py:class:`FunctionTransformer<sklearn.preprocessing.FunctionTransformer>`.
        We can also use other transformers that only learn the input shape at
        training time, such as
        :py:class:`PolynomialFeatures<sklearn.preprocessing.PolynomialFeatures>`.
        Note that these do require us to set `attempt_fit_transformer=True`. If left
        unspecified, this defaults to
        `FunctionTransformer(lambda x: np.hstack((x, x ** 2)), validate=False)`,
        which generates point-wise quadratic interactions.

    attempt_fit_transformer : bool, optional
        Whether to attempt to pre-fit the transformer. Note: this is done on
        only one data point. This works for transformers that only require the
        input shape and/or dtype for fitting. In other words, this will *not*
        work for more sophisticated transformers that require batch
        aggregations.

    regressor_kwargs : keyword arguments
        Keyword arguments for
        :py:class:`SGDRegressor<sklearn.linear_model.SGDRegressor>`.
        The default value for `learning_rate` is set to `'constant'`.

    """
    def __init__(self, env, transformer=None, attempt_fit_transformer=False,
                 **regressor_kwargs):

        # prepare model
        kwargs = {'learning_rate': 'constant'}  # defaults
        kwargs.update(regressor_kwargs)  # override defaults
        self.regressor_kwargs = kwargs
        regressor = SGDRegressor(**kwargs)
        if transformer is None:
            transformer = FunctionTransformer(
                lambda x: np.hstack((x, x ** 2)), validate=False)

        super(LinearV, self).__init__(
            env=env, regressor=regressor, transformer=transformer,
            model_type=0,
            attempt_fit_transformer=attempt_fit_transformer)

    def __repr__(self):
        return (
            "LinearV(env={env},\n"
            "        transformer={transformer},\n"
            "        model_type={model_type},\n"
            "        attempt_fit_transformer={attempt_fit_transformer},\n"
            "        {regressor_kwargs_})"
            .format(
                regressor_kwargs_=", ".join(
                    "{}={}".format(k, v)
                    for k, v in self.regressor_kwargs.items()),
                **self.__dict__
            ))


class LinearQ(GenericQ):
    """
    A linear function approximator for a state-action value function
    :math:`Q(s,a)` based on scikit-learn's
    :py:class:`SGDRegressor<sklearn.linear_model.SGDRegressor>`.

    Parameters
    ----------
    env : gym environment spec
        This is used to get information about the shape of the observation
        space and action space.

    transformer : sklearn transformer, optional
        Unfortunately, there's no support for out-of-core fitting of
        transformers in scikit-learn. We can, however, use stateless
        transformers such as
        :py:class:`FunctionTransformer<sklearn.preprocessing.FunctionTransformer>`.
        We can also use other transformers that only learn the input shape at
        training time, such as
        :py:class:`PolynomialFeatures<sklearn.preprocessing.PolynomialFeatures>`.
        Note that these do require us to set `attempt_fit_transformer=True`. If left
        unspecified, this defaults to
        `FunctionTransformer(lambda x: np.hstack((x, x ** 2)), validate=False)`,
        which generates point-wise quadratic interactions.

    model_type : {1, 2}
        Whether to model the state-action value function as
        :math:`(s, a)\\mapsto Q(s, a)` (type I) or :math:`s\\mapsto Q(s, .)`
        (type II). The latter returns a vector of values, one entry for each
        possible action.

    attempt_fit_transformer : bool, optional
        Whether to attempt to pre-fit the transformer. Note: this is done on
        only one data point. This works for transformers that only require the
        input shape and/or dtype for fitting. In other words, this will *not*
        work for more sophisticated transformers that require batch
        aggregations.

    state_action_combiner : {'cross', 'concatenate'} or func, optional
        This option only matters if `model_type=1`; it specifies how we choose
        to combine the feature vectors coming from `s` and `a`.
        Here 'cross' means taking a flat cross product using
        :py:func:`numpy.kron`, which gives a 1d-array of length
        `dim_s * dim_a`. This choice is suitable for simple linear models,
        including the table-lookup type models. In contrast, 'concatenate'
        uses :py:func:`numpy.hstack` to return a 1d array of length
        `dim_s + dim_a`. This option is more suitable for non-linear models.

    regressor_kwargs : keyword arguments
        Keyword arguments for
        :py:class:`SGDRegressor<sklearn.linear_model.SGDRegressor>`.
        The default value for `learning_rate` is set to `'constant'`.

    """
    def __init__(self, env, transformer=None, model_type=1,
                 attempt_fit_transformer=False, state_action_combiner='cross',
                 **regressor_kwargs):

        # prepare model
        kwargs = {'learning_rate': 'constant'}  # defaults
        kwargs.update(regressor_kwargs)  # override defaults
        self.regressor_kwargs = kwargs
        regressor = SGDRegressor(**kwargs)
        if transformer is None:
            transformer = FunctionTransformer(
                lambda x: np.hstack((x, x ** 2)), validate=False)

        super(LinearQ, self).__init__(
            env=env, regressor=regressor, transformer=transformer,
            model_type=model_type,
            attempt_fit_transformer=attempt_fit_transformer,
            state_action_combiner=state_action_combiner)

    def __repr__(self):
        return (
            "LinearQ(env={env},\n"
            "        transformer={transformer},\n"
            "        model_type={model_type},\n"
            "        attempt_fit_transformer={attempt_fit_transformer},\n"
            "        state_action_combiner={state_action_combiner},\n"
            "        {regressor_kwargs_})"
            .format(
                regressor_kwargs_=", ".join(
                    "{}={}".format(k, v)
                    for k, v in self.regressor_kwargs.items()),
                **self.__dict__
            ))
