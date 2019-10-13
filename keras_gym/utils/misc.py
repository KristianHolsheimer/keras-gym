import os
import time
import logging

from PIL import Image


__all__ = (
    'enable_logging',
    'generate_gif',
    'get_env_attr',
    'get_transition',
    'has_env_attr',
    'is_policy',
    'is_qfunction',
    'is_vfunction',
    'render_episode',
    'set_tf_loglevel',
)


def enable_logging(silence_tf_logging=True):
    """

    Enable logging output.

    This runs the following lines of code:

    .. code:: python

        import logging
        logging.basicConfig(level=logging.INFO)
        if silence_tf_logging:
            set_tf_loglevel(logging.ERROR)  # another helper function


    Parameters
    ----------
    silence_tf_logging : bool, optional

        Whether to silence Tensorflow logging.

    """
    logging.basicConfig(level=logging.INFO)
    if silence_tf_logging:
        set_tf_loglevel(logging.ERROR)


def get_transition(env):
    """
    Generate a transition from the environment.

    This basically does a single step on the environment
    and then closes it.

    Parameters
    ----------
    env : gym environment

        A gym environment.

    Returns
    -------
    s, a, r, s_next, a_next, done, info : tuple

        A single transition. Note that the order and the number of items
        returned is different from what ``env.reset()`` return.

    """
    try:
        s = env.reset()
        a = env.action_space.sample()
        a_next = env.action_space.sample()
        s_next, r, done, info = env.step(a)
        return s, a, r, s_next, a_next, done, info
    finally:
        env.close()


def render_episode(env, policy, step_delay_ms=0):
    """
    Run a single episode with env.render() calls with each time step.

    Parameters
    ----------
    env : gym environment

        A gym environment.

    policy : callable

        A policy objects that is used to pick actions: ``a = policy(s)``.

    step_delay_ms : non-negative float

        The number of milliseconds to wait between consecutive timesteps. This
        can be used to slow down the rendering.

    """
    s = env.reset()
    env.render()

    for t in range(int(1e9)):
        a = policy(s)
        s_next, r, done, info = env.step(a)

        env.render()
        time.sleep(step_delay_ms / 1e3)

        if done:
            break

        s = s_next

    time.sleep(5 * step_delay_ms / 1e3)
    env.close()


def has_env_attr(env, attr, max_depth=100):
    """
    Check if a potentially wrapped environment has a given attribute.

    Parameters
    ----------
    env : gym environment

        A potentially wrapped environment.

    attr : str

        The attribute name.

    max_depth : positive int, optional

        The maximum depth of wrappers to traverse.

    """
    e = env
    for i in range(max_depth):
        if hasattr(e, attr):
            return True
        if not hasattr(e, 'env'):
            break
        e = e.env

    return False


def get_env_attr(env, attr, default='__ERROR__', max_depth=100):
    """
    Get the given attribute from a potentially wrapped environment.

    Note that the wrapped envs are traversed from the outside in. Once the
    attribute is found, the search stops. This means that an inner wrapped env
    may carry the same (possibly conflicting) attribute. This situation is
    *not* resolved by this function.

    Parameters
    ----------
    env : gym environment

        A potentially wrapped environment.

    attr : str

        The attribute name.

    max_depth : positive int, optional

        The maximum depth of wrappers to traverse.

    """
    e = env
    for i in range(max_depth):
        if hasattr(e, attr):
            return getattr(e, attr)
        if not hasattr(e, 'env'):
            break
        e = e.env

    if default == '__ERROR__':
        raise AttributeError("env is missing attribute: {}".format(attr))

    return default


def generate_gif(env, policy, filepath, resize_to=None, duration=50):
    """
    Store a gif from the episode frames.

    Parameters
    ----------
    env : gym environment

        The environment to record from.

    policy : keras-gym policy object

        The policy that is used to take actions.

    filepath : str

        Location of the output gif file.

    resize_to : tuple of ints, optional

        The size of the output frames, ``(width, height)``. Notice the
        ordering: first **width**, then **height**. This is the convention PIL
        uses.

    duration : float, optional

        Time between frames in the animated gif, in milliseconds.

    """
    logger = logging.getLogger('generate_gif')

    # collect frames
    frames = []
    s = env.reset()
    for t in range(env.spec.max_episode_steps or 10000):
        a = policy(s)
        s_next, r, done, info = env.step(a)

        # store frame
        frame = info.get('s_orig', [s])[0]
        frame = Image.fromarray(frame)
        frame = frame.convert('P', palette=Image.ADAPTIVE)
        if resize_to is not None:
            if len(resize_to) != 2:
                raise TypeError("expected a tuple of size 2, resize_to=(w, h)")
            frame = frame.resize(resize_to)

        frames.append(frame)

        if done:
            break

        s = s_next

    # store last frame
    frame = info.get('s_next_orig', [s_next])[0]
    frame = Image.fromarray(frame)
    frame = frame.convert('P', palette=Image.ADAPTIVE)
    if resize_to is not None:
        frame = frame.resize(resize_to)
    frames.append(frame)

    # generate gif
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    frames[0].save(
        fp=filepath, format='GIF', append_images=frames[1:], save_all=True,
        duration=duration, loop=0)

    logger.info("recorded episode to: {}".format(filepath))


def is_vfunction(obj):
    """
    Check whether an object is a :term:`state value function`, or V-function.

    Parameters
    ----------
    obj

        Object to check.

    Returns
    -------
    bool

        Whether ``obj`` is a V-function.

    """
    # import at runtime to avoid circular dependence
    from ..function_approximators.value_v import V
    return isinstance(obj, V)


def is_qfunction(obj, qtype=None):
    """

    Check whether an object is a :term:`state-action value function <type-I
    state-action value function>`, or Q-function.

    Parameters
    ----------
    obj

        Object to check.

    qtype : 1 or 2, optional

        Check for specific Q-function type, i.e. :term:`type-I <type-I
        state-action value function>` or :term:`type-II <type-II state-action
        value function>`.

    Returns
    -------
    bool

        Whether ``obj`` is a (type-I/II) Q-function.

    """
    # import at runtime to avoid circular dependence
    from ..function_approximators.value_q import QTypeI, QTypeII

    if qtype is None:
        return isinstance(obj, (QTypeI, QTypeII))
    elif qtype in (1, 1., '1', 'i', 'I'):
        return isinstance(obj, QTypeI)
    elif qtype in (2, 2., '2', 'ii', 'II'):
        return isinstance(obj, QTypeII)
    else:
        raise ValueError("unexpected qtype: {}".format(qtype))


def is_policy(obj, check_updateable=False):
    """

    Check whether an object is an :term:`(updateable) policy <updateable
    policy>`.

    Parameters
    ----------
    obj

        Object to check.

    check_updateable : bool, optional

        If the obj is a policy, also check whether or not the policy is
        updateable.

    Returns
    -------
    bool

        Whether ``obj`` is a (updateable) policy.

    """
    # import at runtime to avoid circular dependence
    from ..policies.base import BasePolicy
    from ..function_approximators.base import BaseUpdateablePolicy

    if check_updateable:
        return isinstance(obj, BaseUpdateablePolicy)
    return isinstance(obj, BasePolicy)


def set_tf_loglevel(level):
    """

    Set the logging level for Tensorflow logger. This also sets the logging
    level of the underlying C++ layer.

    Parameters
    ----------
    level : int

        A logging level as provided by the builtin :mod:`logging` module, e.g.
        ``level=logging.INFO``.

    """
    if level >= logging.FATAL:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    if level >= logging.ERROR:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    if level >= logging.WARNING:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    else:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
    logging.getLogger('tensorflow').setLevel(level)
