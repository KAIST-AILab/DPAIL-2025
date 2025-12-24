# from torch.utils.tensorboard import SummaryWriter
import wandb
import os
import numpy as np
from time import time, sleep
from datetime import timedelta
import matplotlib.pyplot as plt
COLORS = ['r', 'blue', 'g', 'k', 'm']
def make_xy_plot(trajs, name='test.png', goals=[[1, 10], [3, 8], [7, 10], [5, 4], [7, 1]]):
    fig, ax = plt.subplots()
    for traj in trajs:
        traj = np.vstack(traj)
        ax.scatter(traj[:, 0], traj[:, 1])
    for idx, goal in enumerate(goals):
        circle = plt.Circle(goal, 0.25, color=COLORS[idx])
        ax.add_patch(circle)
    plt.xlim(-1, 10)
    plt.ylim(0.5, 10.5)
    # plt.title(f'{len(traj)}-length')
    plt.savefig(name)
    plt.close(fig)
    
class Trainer:

    def __init__(self, env_id, env, env_test, algo, algo_id, log_dir, one_step_control=True, seed=0, num_steps=10**5,
                 eval_interval=10**3, num_eval_episodes=10):
        super().__init__()

        # Env to collect samples
        self.env = env
        self.env_id = env_id

        # Env for evaluation
        self.env_test = env_test

        self.algo = algo
        self.algo_id = algo_id
        self.log_dir = log_dir
        self.one_step_control = one_step_control

        # Log setting
        self.summary_dir = log_dir
        # self.writer = SummaryWriter(log_dir=self.summary_dir)
        self.model_dir = os.path.join(log_dir, 'model')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        # Other parameters
        self.num_steps = num_steps
        self.eval_interval = eval_interval
        self.num_eval_episodes = num_eval_episodes

    def train(self):
        # Time to start training
        self.start_time = time()
        # Episode's timestep
        t = 0
        # Initialize the environment
        state = self.env.reset()

        for step in range(1, self.num_steps + 1):
            # Pass to the algorithm to update state and episode timestep
            state, t = self.algo.step(self.env, state, t, step)

            # Update the algorithm whenever ready
            if self.algo.is_update(step):
                self.algo.update()

            # Evaluate regularly
            if step % self.eval_interval == 0:
                self.evaluate(step)
                if (step > 1000000) and (step % 10*5 == 0):
                    self.algo.save_models(
                        os.path.join(self.model_dir, f'step{step}'))

    def evaluate(self, step, make_plot=True):
        mean_return = 0.0
        mean_episode_length = 0.0
        mean_min_distance = 0.0
        if make_plot: trajs = [[]] * self.num_eval_episodes 
        for epi in range(self.num_eval_episodes):
            state = self.env_test.reset()
            episode_return = 0.0
            done = False
            t = 0
            if self.algo_id == 'infogail':
                code = self.algo.sample_code()[1]
            while (not done):
                if make_plot: trajs[epi].append(state[:2])
                if self.one_step_control:
                    action = self.algo.exploit(np.concatenate([state, code])) \
                        if self.algo_id == "infogail" else self.algo.exploit(state)
                else:
                    if (t % self.algo.act_seq_length == 0):
                        action_sequence = self.algo.explore_action_sequence(state)
                    action = action_sequence[t % self.algo.act_seq_length]
                state, reward, done, info = self.env_test.step(action)
                episode_return += reward
                t += 1
            mean_return += episode_return / self.num_eval_episodes
            mean_episode_length += t / self.num_eval_episodes
            
            if 'goal_0' in info.keys():
                min_distance = 100
                for k in info.keys():
                    min_distance = info[k] if info[k] < min_distance else min_distance
                mean_min_distance += min_distance/self.num_eval_episodes 
        if make_plot : make_xy_plot(trajs, name=os.path.join(self.log_dir, f'{step}.png'))
                
        print(f'Num steps: {step:<6}   '
              f'Return/Succss_Rate: {mean_return:<5.1f}   '
              f'Mean min Dist: {mean_min_distance:<5.1f}'
              f'Time: {self.time}')
        
        wandb.log({
            'return': mean_return,
            'mean_episode_length': mean_episode_length,
            'mean_min_distancs': mean_min_distance,
            })

    @property
    def time(self):
        return str(timedelta(seconds=int(time() - self.start_time)))
