About
=====

When I first started to play around with reinforcement learning, I noticed that
I was writing the same boiler-plate code over and over again. Moreover, I found
myself implementing gradient descent, adagrad, momentum etc. from scratch every
time I just wanted to try out a simple TD(0) type algorithm. Meanwhile, the
state of plain vanilla supervised learning has evolved to a point where, from a
practitioner's point of view, it's just a matter of importing the right
library. So why wouldn't we do the same for reinforcement learning? Of course
there's the amazing `gym <https://gym.openai.com/>`_ package, which has created a
standard for dealing with MDPs. From the side of specific implementation of
policies and algorithms, though, such a standard hadn't been reached. So that's
the reason why I started this project.

To be a bit more precise, my goal was to make the process of implementing
algorithm is as simple as possible and to make it completely separate from the
code that does function approximation.

If you have any bugs, questions or suggestions, please feel free to open an
issue on `github <https://github.com/KristianHolsheimer/keras-gym/>`_.

-- Kristian Holsheimer
