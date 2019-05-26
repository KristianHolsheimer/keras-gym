from gym.envs.toy_text.frozen_lake import FrozenLakeEnv, RIGHT, DOWN

from .special import UserInputPolicy


class MockInputFunction:
    def __init__(self, return_value=None):
        self.return_value = return_value
        self._orig_input_fn = __builtins__['input']

    def _mock_input_fn(self, prompt):
        print(prompt + str(self.return_value))
        return self.return_value

    def __enter__(self):
        __builtins__['input'] = self._mock_input_fn

    def __exit__(self, type, value, traceback):
        __builtins__['input'] = self._orig_input_fn


class TestUserInputPolicy:

    def test_expected(self):
        env = FrozenLakeEnv(is_slippery=False)
        policy = UserInputPolicy(env)

        s = env.reset()
        env.render()

        for i in [RIGHT, RIGHT, DOWN, DOWN, DOWN, RIGHT]:
            with MockInputFunction(return_value=i):
                a = policy(s)

            s, r, done, info = env.step(a)
            env.render()

            if done:
                break
