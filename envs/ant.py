import copy
import gym
import numpy as np
from gym import spaces
from gym.envs.mujoco.ant_v3 import AntEnv

class MultimodalAnt(AntEnv):
    """A multi-modal variant of the Ant environment.

    https://github.com/openai/gym/blob/master/gym/envs/mujoco/ant_v3.py

    """

    def __init__(self, num_modes=4, mode_idx=None):
        self.t = 0
        self.num_modes = num_modes
        self._max_episode_steps = 1000
        self.mode_idx = mode_idx
        super().__init__(exclude_current_positions_from_observation=False)
        # super().__init__()
        

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()
        # contact_force = self.contact_forces.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[2:]

        observations = np.concatenate((position, velocity))

        return observations

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
        xy_position_before = self.get_body_com("torso")[:2].copy()
        self.do_simulation(action, self.frame_skip)
        xy_position_after = self.get_body_com("torso")[:2].copy()

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        ctrl_cost = self.control_cost(action)
        contact_cost = self.contact_cost
        healthy_reward = self.healthy_reward

        forward_reward, position_penalty = 0.0, 0.0
        mode_info = {}
        if self.mode_idx == 0:
            forward_reward = x_velocity
            # position_penalty = np.clip(np.abs(xy_position_after[1]) - 1., 0, None)
        elif self.mode_idx == 1:
            forward_reward = -x_velocity
            # position_penalty = np.clip(np.abs(xy_position_after[1]) - 1., 0, None)
        elif self.mode_idx == 2:
            forward_reward = y_velocity
            # position_penalty = np.clip(np.abs(xy_position_after[0]) - 1., 0, None)
        elif self.mode_idx == 3:
            forward_reward = -y_velocity
            # position_penalty = np.clip(np.abs(xy_position_after[0]) - 1., 0, None)
        # None
        else:
            mode_info.update({
                'mode':
                    {
                        0: x_velocity,
                        1: -x_velocity,
                        2: y_velocity,
                        3: -y_velocity,
                    }
            })
            
        truncated = False
        if self.t == self._max_episode_steps:
            truncated = True

        terminated = self.done
        reward = forward_reward - position_penalty + healthy_reward - ctrl_cost - contact_cost
        observation = self._get_obs()
        info = {
            "reward_forward": forward_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_contact": -contact_cost,
            "reward_survive": healthy_reward,
            "reward_absvel": -position_penalty,
            "x_position": xy_position_after[0],
            "y_position": xy_position_after[1],
            "distance_from_origin": np.linalg.norm(xy_position_after, ord=2),
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            "forward_reward": forward_reward,
        }
        done = terminated or truncated
        info.update(mode_info)
        # self.renderer.render_step()
        return observation, reward, done, info

    @property
    def expert_scores(self):
        return [4725, 5050, 5099, 5315]

