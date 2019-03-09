Function Approximator Wrappers
==============================

These are compatibility wrappers that allow you to use e.g. scikit-learn
instead of Keras function approximator. A simple example might be:

.. code:: python

    from keras_gym.value_functions import GenericQ
    from keras_gym.wrappers import SklearnModelWrapper

    from sklearn.linear_model import SGDRegressor
    from sklearn.preprocessing import FunctionTransformer


    # the environment
    env = gym.make('CartPole-v0')

    # define sklearn model for approximating Q-function
    estimator = SGDRegressor(eta0=0.08, learning_rate='constant')
    transformer = FunctionTransformer(
        lambda x: np.hstack([x, x ** 2]), validate=False)

    # make it look like a Keras model
    model = SklearnModelWrapper(estimator, transformer)

    # define your Q-function
    Q = GenericQ(env, model)

    # write the rest of the code
    ...



.. toctree::
    :maxdepth: 2
    :glob:

    *
