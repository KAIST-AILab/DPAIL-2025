import gym
from envs import MultimodalEnvs
# gym.logger.set_level(40)


def make_env(env_id, mode_idx=None):
    # try:
    env = MultimodalEnvs[env_id](mode_idx=mode_idx)
    return NormalizedEnv(env)
    # except:
    # return NormalizedEnv(gym.make(env_id))


class NormalizedEnv(gym.Wrapper):

    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self._max_episode_steps = env._max_episode_steps

        self.scale = env.action_space.high
        self.action_space.high /= self.scale
        self.action_space.low /= self.scale

    def step(self, action):
        return self.env.step(action * self.scale)
