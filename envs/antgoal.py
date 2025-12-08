import copy
import gym
import numpy as np
from gym import spaces
from gym.envs.mujoco.ant_v3 import AntEnv

class MultimodalAntGoal(AntEnv):
    """A multi-modal variant of the Ant environment.

    https://github.com/openai/gym/blob/master/gym/envs/mujoco/ant_v3.py

    """

    def __init__(self, num_modes=8, mode_idx=None):
        self.t = 0
        self.num_modes = num_modes
        self._max_episode_steps = 1000
        self.mode_idx = mode_idx
        self.goals = np.array([
                               [20, 0], [0, 20], 
                               [-20, 0], [0, -20],
                               [14.14, 14.14], [-14.14, 14.14],
                               [-14.14, -14.14], [14.14, -14.14],
                               ])
        self.found_goal = False
        super().__init__(exclude_current_positions_from_observation=False)
        # self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(376,), dtype=np.float64)

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()
        contact_force = self.contact_forces.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[2:]

        observations = np.concatenate((position, velocity))

        return observations
        
    def reset_model(self):
        self.t = 0
        self.found_goal = False
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
        
        mode_info = {}
        prev_dist, dist = 0, 0
        if self.mode_idx is None:
            forward_rewards = {}
            for i in range(8):
                prev_dist = np.linalg.norm((xy_position_before[:2] - self.goals[i])) 
                dist = np.linalg.norm((xy_position_after[:2] - self.goals[i])) 
                forward_rewards.update({
                    i: prev_dist - dist
                })
            mode_info.update({'mode': forward_rewards})
        else:
            prev_dist = np.linalg.norm((xy_position_before[:2] - self.goals[self.mode_idx])) 
            dist = np.linalg.norm((xy_position_after[:2] - self.goals[self.mode_idx])) 
        forward_reward = prev_dist - dist #/ self.dt
        # forward_reward = 1 - (dist/self.distance)
        ctrl_cost = self.control_cost(action)
        contact_cost = self.contact_cost

        healthy_reward = self.healthy_reward
        truncated = False
        if self.t == self._max_episode_steps:
            truncated = True

        terminated = self.done
        success = 0.
        if dist < 0.5:
            if not self.found_goal:
                print("Success")
                success = 1.
            self.found_goal = True
        reward = 10 * forward_reward + healthy_reward - ctrl_cost - contact_cost
        observation = self._get_obs()
        info = {
            "reward_forward": forward_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_contact": -contact_cost,
            "reward_survive": healthy_reward,
            "reward_absvel": success,
            "x_position": xy_position_after[0],
            "y_position": xy_position_after[1],
            "distance_from_origin": np.linalg.norm(xy_position_after, ord=2),
            "forward_reward": forward_reward,
        }
        done = terminated or truncated
        info.update(mode_info)
        # self.renderer.render_step()
        return observation, reward, done, info

    @property
    def expert_scores(self):
        return [19.15, 19.22, 19.58, 18.88, 19.46, 19.31, 19.54, 19.82]

