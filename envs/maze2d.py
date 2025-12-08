import copy
import gym
import numpy as np
from gym import spaces
from d4rl.pointmaze import MazeEnv, MEDIUM_MAZE, LARGE_MAZE


class MultiGoalMaze2dMedium(MazeEnv):
    """A multi-goal of the maze2d-medium-v1 environment.

    """
    def __init__(self, num_goals=3, mode_idx=0):
        self.goals = np.array([[1.0, 6.0], [6.0, 5.0], [6.0, 1.0]])
        self.t = 0
        self._max_episode_steps = 600
        super().__init__(maze_spec=MEDIUM_MAZE, reward_type='sparse',reset_target=False)
        self.num_goals = num_goals
        self.goal_idx = None
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float64)

    def reset_model(self):
        self.t = 0
        idx = 0 # (1, 1)
        reset_location = np.array(self.empty_and_goal_locations[idx]).astype(self.observation_space.dtype)
        qpos = reset_location + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        if self.reset_target:
            self.set_target()
        return self._get_obs()

    def step(self, action):
        self.t += 1
        next_state, _, done, _ = super().step(action)
        dist = np.linalg.norm((next_state[None, :2] - self.goals), axis=-1)
        check_goals = (dist < 0.25)
        reward = check_goals.any()
        info = {
            'goal_0' : dist[0],
            'goal_1' : dist[1],
            'goal_2' : dist[2],
        }
        if reward: done = True
        if self.t == self._max_episode_steps: done = True
        
        return next_state, reward, done, info

    def check_goals(self, state):
        return (np.linalg.norm((state[None, :2] - self.goals), axis=-1) < 0.25)

class MultiGoalMaze2dLarge(MazeEnv):
    """A multi-goal of the maze2d-large-v1 environment.
    """
    def __init__(self, num_goals=5, mode_idx=0):
        self.goals = np.array([[1.0, 10.0], [3.0, 8.0], [7.0, 10.0], [5.0, 4.0], [7.0, 1.0]])
        self.t = 0
        self._max_episode_steps = 800
        super().__init__(maze_spec=LARGE_MAZE, reward_type='sparse',reset_target=False)
        self.num_goals = num_goals
        self.mode_idx = None
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float64)

    def reset_model(self):
        self.t = 0
        idx = 0 # (1, 1)
        reset_location = np.array(self.empty_and_goal_locations[idx]).astype(self.observation_space.dtype)
        qpos = reset_location + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        if self.reset_target:
            self.set_target()
        return self._get_obs()

    def step(self, action):
        self.t += 1
        next_state, _, done, _ = super().step(action)
        dist = np.linalg.norm((next_state[None, :2] - self.goals), axis=-1)
        check_goals = (dist < 0.25)
        reward = check_goals.any()
        info = {
            'goal_0' : dist[0],
            'goal_1' : dist[1],
            'goal_2' : dist[2],
            'goal_3' : dist[3],
            'goal_4' : dist[4],
            
        }
        if reward: done = True
        if self.t == self._max_episode_steps: done = True
        
        return next_state, reward, done, info

    def check_goals(self, state):
        return (np.linalg.norm((state[None, :2] - self.goals), axis=-1) < 0.25)
        
if __name__=="__main__":
    env = MultiGoalMaze2dLarge()
    
    for epi in range(10):
        init_pose = env.reset()
        done = False
        t = 0
        while not done:
            n_s, r, done, _ = env.step(env.action_space.sample())
            t += 1
        print(init_pose)
        # print(t)