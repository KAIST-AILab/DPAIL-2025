import copy
import gym
import numpy as np
from gym import spaces
from gym.envs.mujoco.walker2d_v3 import Walker2dEnv


class MultimodalWalker(Walker2dEnv):
    """A multi-modal variant of the Walker-2D environment.
    https://github.com/openai/gym/blob/master/gym/envs/mujoco/walker2d_v3.py

    """
    def __init__(self, num_modes=2, mode_idx=None):
        self.num_modes = num_modes
        self._max_episode_steps = 1000
        self.t = 0
        self.mode_idx = mode_idx
        super().__init__(exclude_current_positions_from_observation=False)

    def reset_model(self):
        self.t = 0
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv
        )

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation

    def step(self, action):
        self.t += 1
        x_position_before = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.data.qpos[0]
        y_position_after = self.data.qpos[1]
        x_velocity = (x_position_after - x_position_before) / self.dt

        ctrl_cost = self.control_cost(action)
        healthy_reward = self.healthy_reward

        terminated = self.done
        if terminated:
            healthy_reward = -100.0
        mode_info = {}
        forward_reward = 0.0
        abs_vel_reward = 0.0  # Prevent standing still
        if self.mode_idx == 0:
            forward_reward = self._forward_reward_weight * x_velocity
            abs_vel_reward = self._forward_reward_weight * np.fabs(x_velocity)
        elif self.mode_idx == 1:
            forward_reward = self._forward_reward_weight * -x_velocity
            abs_vel_reward = self._forward_reward_weight * np.fabs(x_velocity)
        # elif self.mode_idx == 2:
        #     forward_reward = self._forward_reward_weight * -np.fabs(x_velocity)
        #     abs_vel_reward = self._forward_reward_weight * np.fabs(x_velocity)
        # when mode_idx=None
        else:
            mode_info.update({
                'mode':
                    {
                        0: self._forward_reward_weight*x_velocity,
                        1: -self._forward_reward_weight*x_velocity,
                    }
            })
            
        observation = self._get_obs()

        truncated = False
        if self.t == self._max_episode_steps:
            truncated = True

        reward = forward_reward + healthy_reward - ctrl_cost
        info = {"x_position": x_position_after,
                "y_position": y_position_after,
                "reward_forward": forward_reward,
                "reward_ctrl": -ctrl_cost,
                "reward_survie": healthy_reward,
                "reward_absvel": -abs_vel_reward}
        done = terminated or truncated
        info.update(mode_info)
        return observation, reward, done, info
    
    @property
    def expert_scores(self):
        return [3909, 3198]
