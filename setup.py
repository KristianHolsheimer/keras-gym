#!/usr/bin/env python3
# flake8: noqa
import re
import os
import setuptools


pwd = os.path.dirname(__file__)

install_requires = []
with open(os.path.join(pwd, 'requirements.txt')) as f:
    # check if tensorflow-gpu is installed
    try:
        from pip.commands.freeze import freeze
        gpu = any(re.match(r'^tensorflow-gpu==\d+\.\d+\.\d+.*$', p) for p in freeze())
    except ImportError:
        gpu = False

    for line in f:
        line = line.strip()
        if line and not line.startswith('#'):
            # don't install 'tensorflow' alongside 'tensorflow-gpu'
            if gpu and re.match(r'^tensorflow(?:==|>=|<=|>|<)\d+\.\d+\.\d+.*$', line):
                line = line.replace('tensorflow', 'tensorflow-gpu')
            install_requires.append(line)

with open(os.path.join(pwd, 'keras_gym', '__init__.py')) as f:
    version = re.search(r'__version__ \= \'(\d+\.\d+\.\d+)\'', f.read())
    assert version is not None, "can't parse __version__ from __init__.py"
    version = version.groups()[0]
    assert len(version.split('.')) == 3, "bad version spec"
    majorminor = version.rsplit('.', 1)[0]


dev_status = {
    '0.1': 'Development Status :: 1 - Planning',          # v0.1 - skeleton
    '0.2': 'Development Status :: 2 - Pre-Alpha',         # v0.2 - some basic functionality
    '0.3': 'Development Status :: 3 - Alpha',             # v0.3 - most functionality
    '0.4': 'Development Status :: 4 - Beta',              # v0.4 - most functionality + doc
    '1.0': 'Development Status :: 5 - Production/Stable', # v1.0 - most functionality + doc + test  # noqa
    '2.0': 'Development Status :: 6 - Mature',            # v2.0 - new functionality?
}


long_description = """
keras-gym: Plug-n-play reinforcement learning in python with OpenAI Gym and
Keras.

For full documentation, go to:

    https://keras-gym.readthedocs.io

You can find the source code at:

    https://github.com/KristianHolsheimer/keras-gym

"""

# main setup kw args
setup_kwargs = {
    'name': 'keras-gym',
    'version': version,
    'description': "Plug-n-play reinforcement learning with OpenAI Gym and Keras",
    'long_description': long_description,
    'author': 'Kristian Holsheimer',
    'author_email': 'kristian.holsheimer@gmail.com',
    'url': '',
    'license': 'MIT',
    'install_requires': install_requires,
    'classifiers': [
        dev_status[majorminor],
        'Environment :: Other Environment',
        'Framework :: Jupyter',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    'zip_safe': True,
    'packages': setuptools.find_packages(exclude=['test_*.py']),
}


if __name__ == '__main__':
    setuptools.setup(**setup_kwargs)
