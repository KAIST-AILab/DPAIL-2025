# from torch.utils.tensorboard import SummaryWriter
import wandb
import os
import numpy as np
from time import time
from datetime import timedelta
from imitation.utils import entropy, evaluate_entropy_for_sa
from collections import deque

class Trainer:

    def __init__(self, env_id, mode_list, env, env_test, algo, algo_id, log_dir, one_step_control=True, seed=0, num_steps=10**5, eval_interval=10**3, num_eval_episodes=5):
        super().__init__()

        # Env to collect samples.
        self.env = env
        self.env_id = env_id

        # Env for evaluation.
        self.env_test = env_test

        self.algo = algo
        self.algo_id = algo_id
        self.log_dir = log_dir
        self.one_step_control = one_step_control

        # Log setting.
        self.summary_dir = log_dir
        self.model_dir = os.path.join(log_dir, 'model')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        # Other parameters.
        self.num_steps = num_steps
        self.eval_interval = eval_interval
        self.num_eval_episodes = num_eval_episodes
        self.mode_mask = np.zeros(self.env_test.num_modes)
        for mode in mode_list:
            self.mode_mask[mode] = 1
        

    def train(self):
        # Time to start training
        self.start_time = time()
        # Episode's timestep
        t = 0
        # Initialize the environment
        state = self.env.reset()
        step = -1
        if self.algo.is_update(step):
            print("Start init update")
            self.algo.update()
            [print("End init_update")]
        for step in range(1, self.num_steps + 1):
            # Pass to the algorithm to update state and episode timestep
            state, t = self.algo.step(self.env, state, t, step)

            # Update the algorithm whenever ready
            if self.algo.is_update(step):
                self.algo.update()

            # Evaluate regularly
            if step % self.eval_interval == 0:
                if (step > (self.num_steps - self.eval_interval * 5)):
                    self.final_evaluate(step)
                    self.algo.save_models(os.path.join(self.model_dir, f'step{step}'))
                    print(f'Save Model Step {step}')
                else:
                    self.evaluate(step)

    def evaluate(self, step):
        mean_success = 0.0
        mean_episode_length = 0.0
        success_mode_idx_list, acc_rewards, trajs = [], [], []
        for epi in range(self.num_eval_episodes):
            state = self.env_test.reset()
            episode_returns = np.zeros(self.env_test.num_modes)
            done = False
            t = 0
            traj = []
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
                for (k, v) in info['mode'].items():
                    episode_returns[k] += v
                traj.append(np.array([info['x_position'], info['y_position']]))
                t += 1
            trajs.append(np.vstack(traj))
            success, acc_reward, success_mode_idx = self.calculate_success_rate(episode_returns)
            mean_success += success / self.num_eval_episodes
            mean_episode_length += t / self.num_eval_episodes
            success_mode_idx_list.append(success_mode_idx)
            acc_rewards.append(acc_reward)
        ent = entropy(success_mode_idx_list)
        print(f'Num steps: {step:<6}   '
              f'Success: {mean_success:<5.1f}   '
              f'Ent : {ent:<5.1f}   '
              f'Time: {self.time}')
        print(success_mode_idx_list)
        print(acc_rewards)
        wandb.log({
            'success': mean_success,
            'entropy' : ent,
            'mean_episode_length': mean_episode_length,
            })
    
    def final_evaluate(self, step):
        total_steps = 50000
        success_list, success_mode_idx_list, acc_rewards = deque(maxlen=50), deque(maxlen=50), deque(maxlen=50)
        trajs = []
        state = self.env_test.reset()
        episode_returns = np.zeros(self.env_test.num_modes)
        t = 0
        for s in range(total_steps):
            if ((t == 0) and (self.algo_id == 'infogail')):
                code = self.algo.sample_code()[1]

            if self.one_step_control:
                action = self.algo.exploit(np.concatenate([state, code])) \
                    if self.algo_id == "infogail" else self.algo.exploit(state)
            else:
                if (t % self.algo.act_seq_length == 0):
                    action_sequence = self.algo.explore_action_sequence(state)
                action = action_sequence[t % self.algo.act_seq_length]
            
            state, _, done, info = self.env_test.step(action)
            
            for (k, v) in info['mode'].items():
                episode_returns[k] += v
            trajs.append(np.array([info['x_position'], info['y_position']]))
            t += 1
            
            if done:            
                success, acc_reward, success_mode_idx = self.calculate_success_rate(episode_returns)
                success_list.append(success)
                success_mode_idx_list.append(success_mode_idx)
                acc_rewards.append(acc_reward)
            
                state = self.env_test.reset()
                episode_returns = np.zeros(self.env_test.num_modes)
                done = False
                t = 0
        mean_success = np.mean(success_list)
        task_ent = entropy(success_mode_idx_list)
        sa_ent = evaluate_entropy_for_sa(np.vstack(trajs), env_id=self.env_id)[0]
        print(f'Num steps: {step:<6}   '
              f'EVAL Success: {mean_success:<5.1f}   '
              f'EVAL Task_Ent : {task_ent:<5.1f}   '
              f'EVAL SA_Ent : {sa_ent:<5.1f}   '
              f'Time: {self.time}')
        
        wandb.log({
            'final_success': mean_success,
            'final_task_entropy' : task_ent,
            'final_sa_entropy' : sa_ent,
            })
    
    def calculate_success_rate(self, episode_returns):
        episode_returns = episode_returns * self.mode_mask
        returns = episode_returns.max()
        mode_idx = episode_returns.argmax()
        max_score = self.env_test.expert_scores[mode_idx]
        return returns / max_score, returns, mode_idx
        
    @property
    def time(self):
        return str(timedelta(seconds=int(time() - self.start_time)))
