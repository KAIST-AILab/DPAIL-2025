from tqdm import tqdm
import numpy as np
import torch, math
import matplotlib.pyplot as plt
from ..buffer import Buffer

class EMA():
    '''
        empirical moving average
    '''
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
    
    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)
    
    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def soft_update(target, source, tau):
    for t, s in zip(target.parameters(), source.parameters()):
        t.data.mul_(1.0 - tau)
        t.data.add_(tau * s.data)


def disable_gradient(network):
    for param in network.parameters():
        param.requires_grad = False


def add_random_noise(action, std):
    action += np.random.randn(*action.shape) * std
    return action.clip(-1.0, 1.0)


def collect_demo(env, algo, buffer_size, device, std, p_rand, seed=0):
    env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    buffer = Buffer(
        buffer_size=buffer_size,
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        device=device
    )

    total_return = 0.0
    num_episodes = 0

    state = env.reset()
    t = 0
    episode_return = 0.0

    for _ in tqdm(range(1, buffer_size + 1)):
        t += 1

        if np.random.rand() < p_rand:
            action = env.action_space.sample()
        else:
            action = algo.exploit(state)
            action = add_random_noise(action, std)

        next_state, reward, done, info = env.step(action)
        mask = False if t == env._max_episode_steps else done
        terminated = done # episode_end flag
        buffer.append(state, action, reward, mask, next_state, terminated=terminated)
        episode_return += info["reward_forward"]

        if done:
            num_episodes += 1
            total_return += episode_return
            print(f"Epi {num_episodes}: {episode_return}")
            state = env.reset()
            t = 0
            episode_return = 0.0
            
        state = next_state
    if not done:
        num_episodes +=1
        print(f"The last episode is ended in {t} step")
        buffer.terminated[-1] = float(True)
    print(f"Num episode {num_episodes} should be same buffer episodes {buffer.terminated.sum()}")
    print(f"Sum of dones {buffer.dones.sum()}")
    print(f'Mean return of the expert is {total_return / num_episodes}')
    return buffer


class RunningMeanStd(object):
    def __init__(self, epsilon=1e-4, shape=()):
        """
        Calculate the running mean and std of a data stream
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
        :param epsilon: helps with arithmetic issues
        :param shape: the shape of the data stream's output
        """
        self.mean = np.zeros(shape, np.float64)
        self.var = np.ones(shape, np.float64)
        self.count = epsilon

    def update(self, arr: np.ndarray) -> None:
        batch_mean = np.mean(arr, axis=0)
        batch_var = np.var(arr, axis=0)
        batch_count = arr.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int) -> None:
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = m_2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count


class RunningMeanNormalizer(RunningMeanStd):
    def __init__(self, input_dim, epsilon=1e-4, clip_obs=10.0, env="Ant-v3"):
        super().__init__(shape=input_dim)
        self.epsilon = epsilon
        self.clip_obs = clip_obs

    def normalize(self, input):
        return np.clip(
            (input - self.mean) / np.sqrt(self.var + self.epsilon),
            -self.clip_obs, self.clip_obs)

    def normalize_torch(self, input, device):
        mean_torch = torch.tensor(
            self.mean, device=device, dtype=torch.float32)
        std_torch = torch.sqrt(torch.tensor(
            self.var + self.epsilon, device=device, dtype=torch.float32))
        return torch.clamp(
            (input - mean_torch) / std_torch, -self.clip_obs, self.clip_obs)

    def update_normalizer(self, rollouts, expert_loader):
        policy_data_generator = rollouts.feed_forward_generator_amp(
            None, mini_batch_size=expert_loader.batch_size)
        expert_data_generator = expert_loader.dataset.feed_forward_generator_amp(
                expert_loader.batch_size)

        for expert_batch, policy_batch in zip(expert_data_generator, policy_data_generator):
            self.update(
                torch.vstack(tuple(policy_batch) + tuple(expert_batch)).cpu().numpy())
        ##
        self.mean[:2] = 0.
        self.var[:2] = 100.


class Normalizer:
    '''
        parent class, subclass by defining the `normalize` and `unnormalize` methods
    '''

    def __init__(self, X):
        self.X = X.astype(np.float32)
        self.mins = X.min(axis=0)
        self.maxs = X.max(axis=0)

    def __repr__(self):
        return (
            f'''[ Normalizer ] dim: {self.mins.size}\n    -: '''
            f'''{np.round(self.mins, 2)}\n    +: {np.round(self.maxs, 2)}\n'''
        )

    def __call__(self, x):
        return self.normalize(x)

    def normalize(self, *args, **kwargs):
        raise NotImplementedError()

    def unnormalize(self, *args, **kwargs):
        raise NotImplementedError()



class LimitsNormalizer(Normalizer):
    '''
        maps [ xmin, xmax ] to [ -1, 1 ]
    '''
    def update_normalizer(self, X):
        new_maxs = X.max(axis=0)
        new_mins = X.min(axis=0)
        self.mins = np.minimum(self.mins, new_mins)
        self.maxs = np.maximum(self.maxs, new_maxs)
        
    def normalize_from_torch(self, x, device, eps=1e-7):
        maxs_torch = torch.tensor(self.maxs, device=device, dtype=torch.float32)
        mins_torch = torch.tensor(self.mins, device=device, dtype=torch.float32)
        ## [ 0, 1 ]
        if (self.maxs - self.mins < eps).any():
            eps = np.ones_like(self.maxs) * eps
            x = (x - mins_torch) / np.array([max(i) for i in zip((maxs_torch - mins_torch), eps)])
        else:
            x = (x - mins_torch) / (maxs_torch - mins_torch)
        ## [ -1, 1 ]
        x = 2 * x - 1
        return x

    def normalize(self, x, eps=1e-7):
        ## [ 0, 1 ]
        if (self.maxs - self.mins < eps).any():
            eps = np.ones_like(self.maxs) * eps
            x = (x - self.mins) / np.array([max(i) for i in zip((self.maxs - self.mins), eps)])
        else:
            x = (x - self.mins) / (self.maxs - self.mins)
        ## [ -1, 1 ]
        x = 2 * x - 1
        return x

    def unnormalize(self, x, eps=1e-4):
        '''
            x : [ -1, 1 ]
        '''
        if x.max() > 1 + eps or x.min() < -1 - eps:
            # print(f'[ datasets/mujoco ] Warning: sample out of range | ({x.min():.4f}, {x.max():.4f})')
            x = np.clip(x, -1, 1)

        ## [ -1, 1 ] --> [ 0, 1 ]
        x = (x + 1) / 2.

        return x * (self.maxs - self.mins) + self.min
    
    def unnormalize_from_torch(self, x, device, eps=1e-4):
        maxs_torch = torch.tensor(self.maxs, device=device, dtype=torch.float32)
        mins_torch = torch.tensor(self.mins, device=device, dtype=torch.float32)
        if x.max() > 1 + eps or x.min() < -1 - eps:
            # print(f'[ datasets/mujoco ] Warning: sample out of range | ({x.min():.4f}, {x.max():.4f})')
            x = torch.clip(x, -1, 1) 
        ## [ -1, 1 ] --> [ 0, 1 ]
        x = (x + 1) / 2.
        return x * (maxs_torch - mins_torch) + mins_torch
    
def entropy(labels, base=None):
    """ Computes entropy of label distribution. """

    n_labels = len(labels)

    if n_labels <= 1:
        return 0

    value,counts = np.unique(labels, return_counts=True)
    probs = counts / n_labels
    n_classes = np.count_nonzero(probs)

    if n_classes <= 1:
        return 0

    ent = 0.

    # Compute entropy
    base = math.e if base is None else base
    for i in probs:
        ent -= i * math.log(i, base)

    return ent

def evaluate_entropy_for_sa(trajs, env_id):
    if env_id == "AntGoal-v3":
        cond_red = (2 * trajs[:, 0] > trajs[:, 1])
        cond_blue = (trajs[:, 0] > 2 * trajs[:, 1])
        cond_orange = (-trajs[:, 0] > 2 * trajs[:, 1])
        cond_yellow = (-2 * trajs[:, 0] > trajs[:, 1])
        mode_0 = (cond_blue * (~cond_orange)) * 0
        mode_1 = ((~ cond_red) * (~cond_yellow)) * 1
        mode_2 = ((~ cond_blue) * cond_orange) * 2
        mode_3 = cond_red * cond_yellow * 3
        mode_4 = cond_red * (~ cond_blue) * 4
        mode_5 = ((~ cond_orange) * cond_yellow) * 5
        mode_6 = (cond_blue * (~ cond_red)) * 6
        mode_7 = (cond_orange * (~ cond_yellow)) * 7
        results = mode_0 + mode_1 + mode_2 + mode_3 + mode_4 + mode_5 + mode_6 + mode_7

    elif env_id == "Ant-v3":
        cond_01 = (trajs[:, 0] > trajs[:, 1])
        cond_23 = (-trajs[:, 0] > trajs[:, 1])
        mode_0 = (cond_01 * (~ cond_23)) * 0
        mode_1 = ((~ cond_01) * cond_23) * 1
        mode_2 = ((~ cond_01) * (~ cond_23)) * 2
        mode_3 = (cond_01 * cond_23) * 3
        results = mode_0 + mode_1 + mode_2 + mode_3
    
    else: # "HalfCheetah-v3", "Walker2d-v3"
        results = trajs[:, 0] < 0.
    results = np.array(results, dtype=np.int32)
    return entropy(results), results

def make_xy_plot(trajs, env_id, file_name='test.png'):
    fig, ax = plt.subplots()
    for traj in trajs:
        ax.scatter(traj[:, 0], traj[:, 1])
        # ax.plot(traj[:, 0], traj[:, 1])
    if env_id == "AntGoal-v3":
        plt.xlim(-21.0, 21)
        plt.ylim(-21.0, 21)
    
    elif env_id == "Ant-v3":
        plt.xlim(-200, 200)
        plt.ylim(-200, 200)

    elif env_id == "HalfCheetah-v3":
        plt.xlim(-300, 300)
        plt.ylim(-5, 5)
    
    elif env_id == "Walker2d-v3":
        plt.xlim(-40, 40)
        plt.ylim(-5, 5)
    # plt.title(f'{}-length')
    plt.savefig(file_name)
    plt.close(fig)