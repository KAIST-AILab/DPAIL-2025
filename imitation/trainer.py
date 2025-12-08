# from torch.utils.tensorboard import SummaryWriter
import wandb
import os
import numpy as np
from time import time, sleep
from datetime import timedelta

def analysis_plot(trajs):
    x, y = 0, 0
    for traj in trajs:
        # num_seq, 2
        x_max, y_max = np.max(np.vstack(traj), axis=0)
        x += x_max
        y += y_max
    return {'x_max': x/len(trajs), 
            'y_max': y/len(trajs)}
    
class Trainer:

    def __init__(self, env, env_test, algo, algo_id, log_dir, one_step_control=True, seed=0, num_steps=10**5,
                 eval_interval=10**3, num_eval_episodes=5):
        super().__init__()

        # Env to collect samples.
        self.env = env
        # self.env.seed(seed)

        # Env for evaluation.
        self.env_test = env_test
        # self.env_test.seed(2**31-seed)

        self.algo = algo
        self.algo_id = algo_id
        self.log_dir = log_dir
        self.one_step_control = one_step_control

        # Log setting.
        self.summary_dir = os.path.join(log_dir, 'summary')
        # self.writer = SummaryWriter(log_dir=self.summary_dir)
        self.model_dir = os.path.join(log_dir, 'model')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        # Other parameters.
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
                if (step > 500000) :
                    self.algo.save_models(os.path.join(self.model_dir, f'step{step}'))

        # Wait for the logging to be finished
        # sleep(10)

    def evaluate(self, step):
        mean_return = 0.0
        mean_episode_length = 0.0
        mean_min_distance = 0.0
        mean_forward, mean_absvel = 0.0, 0.0
        for epi in range(self.num_eval_episodes):
            state = self.env_test.reset()
            episode_return, episode_forward, episode_absvel = 0.0, 0.0, 0.0
            done = False
            t = 0
            if self.algo_id == 'infogail':
                code = self.algo.sample_code()[1]
            while (not done):
                if self.one_step_control:
                    action = self.algo.exploit(np.concatenate([state, code])) \
                        if self.algo_id == "infogail" else self.algo.exploit(state)
                else:
                    if (t % self.algo.act_seq_length == 0):
                        action_sequence = self.algo.explore_action_sequence(state)
                    action = action_sequence[t % self.algo.act_seq_length]
                state, reward, done, info = self.env_test.step(action)
                episode_forward += info["reward_forward"]
                episode_absvel += info["reward_absvel"]
                episode_return += reward
                t += 1
            mean_return += episode_return / self.num_eval_episodes
            mean_episode_length += t / self.num_eval_episodes
            mean_forward += episode_forward / self.num_eval_episodes 
            mean_absvel += episode_absvel / self.num_eval_episodes 
                            
        print(f'Num steps: {step:<6}   '
              f'Return: {mean_return:<5.1f}   '
              f'Mean min Dist: {mean_min_distance:<5.1f}'
              f'Time: {self.time}')
        
        wandb.log({
            'return': mean_return,
            'mean_episode_length': mean_episode_length,
            # 'mean_min_distancs': mean_min_distance,
            'mean_forward': mean_forward,
            'mean_absvel': mean_absvel
            })

    @property
    def time(self):
        return str(timedelta(seconds=int(time() - self.start_time)))
