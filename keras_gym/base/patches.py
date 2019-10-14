import re
import os
import tensorflow
import warnings


class PatchException(Exception):
    pass


def run():
    _monkey_patch_tensorflow()


def _monkey_patch_tensorflow():
    """
    See PR: https://github.com/tensorflow/tensorflow/pull/33334

    TODO: Remove this when we can point to a new version of tensorflow that has
    solved this.

    """
    dist_path = os.path.dirname(os.path.dirname(tensorflow.__file__))
    n_missing = 0
    n_updated = 0

    core_py = os.path.join(
        dist_path, 'tensorflow_core', 'python', 'keras', 'layers', 'core.py')

    try:
        with open(core_py, 'r') as r:
            contents = r.read()
            contents, n = re.subn(
                r'if all\(input\_shape\[1\:\]\)\:',
                "if np.all(input_shape[1:]):", contents)
        if n:
            with open(core_py, 'w') as w:
                w.write(contents)
                n_updated += 1

    except FileNotFoundError:
        n_missing += 1

    core_py = os.path.join(
        dist_path, 'tensorflow', 'python', 'keras', 'layers', 'core.py')

    try:
        with open(core_py, 'r') as r:
            contents = r.read()
            contents, n = re.subn(
                r'if all\(input\_shape\[1\:\]\)\:',
                "if np.all(input_shape[1:]):", contents)
        if n:
            with open(core_py, 'w') as w:
                w.write(contents)
                n_updated += 1

    except FileNotFoundError:
        n_missing += 1

    if n_missing == 2:
        warnings.warn(
            "failed to monkey-patch tensorflow; you may run into issues "
            "described in: "
            "https://github.com/tensorflow/tensorflow/pull/33334")

    if n_updated > 0:
        warnings.warn(
            "tensorflow was patched, please rerun or restart your application "
            "to avoid running into issues described in: "
            "https://github.com/tensorflow/tensorflow/pull/33334")
