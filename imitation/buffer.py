import os
import numpy as np
import torch
import random

class SerializedBuffer:

    def __init__(self, directory_path, device, num_modes=1):
        
        self.device = device
        self.indices, self.indices_with_init_state = None, None
        states, actions, rewards, dones, terminated, next_states = [], [], [] ,[], [] ,[]
        pathes = os.listdir(directory_path)
        pathes = random.sample(pathes, num_modes)
        self.mode_list = []
        for path in pathes:
            tmp = torch.load(os.path.join(directory_path, path))
            states.append(tmp['state'].clone())
            actions.append(tmp['action'].clone())
            rewards.append(tmp['reward'].clone())
            dones.append(tmp['done'].clone())
            terminated.append(tmp['terminated'].clone())
            next_states.append(tmp['next_state'].clone())
            print(f"{path} data loaded")
            self.mode_list.append(int(path[4]))
        self.states = torch.vstack(states).to(self.device)
        self.actions = torch.vstack(actions).to(self.device)
        self.rewards = torch.vstack(rewards).to(self.device)
        self.dones = torch.vstack(dones).to(self.device)
        self.terminated = torch.vstack(terminated).to(self.device)
        self.next_states = torch.vstack(next_states).to(self.device)
        self.buffer_size = self._n = self.states.size(0)
        print(f"Buffer size {self.buffer_size}")
    
    def sample(self, batch_size):
        idxes = np.random.randint(low=0, high=self._n, size=batch_size)
        return (
            self.states[idxes],
            self.actions[idxes],
            self.rewards[idxes],
            self.dones[idxes],
            self.next_states[idxes]
        )
    def get(self):
        idxes = slice(0, self.buffer_size)
        return (
            self.states[idxes],
            self.actions[idxes],
            self.rewards[idxes],
            self.dones[idxes],
            self.next_states[idxes],
        )
     ##
    def sample_traj(self, batch_size, horizon):
        if self.indices is None:
            self.start_idxes = [0] + (torch.where(self.terminated)[0] + 1).cpu().numpy().tolist()
            self.path_lengths = []
            start_idx = 0
            for idx in self.start_idxes[1:]:
                path_length = idx - start_idx
                self.path_lengths.append(path_length)
                start_idx += path_length
            if np.sum(self.path_lengths) != self.buffer_size:
                start_idx = self.start_idxes[-1]
                self.path_lengths.append(self.buffer_size - start_idx)
                self.start_idxes.append(0)
            self.indices = self.make_indices(horizon)
        idxes = np.random.randint(low=0, high=len(self.indices), size=batch_size)
        
        # (bs, horizon, dim)
        return (
            torch.stack([self.states[self.indices[i]] for i in idxes]),
            torch.stack([self.actions[self.indices[i]] for i in idxes])
        )
    def make_indices(self, horizon):
        indices = []
        # no padding
        for i, path_length in enumerate(self.path_lengths):
            start_idx = self.start_idxes[i]
            max_start = path_length - horizon + start_idx 
            for start in range(start_idx, max_start):
                end = start + horizon
                indices.append(slice(start, end))
        return indices
    def sample_traj_with_init_state(self, batch_size, horizon):
        if self.indices_with_init_state is None:
            self.start_idxes = [0] + (torch.where(self.terminated)[0] + 1).cpu().numpy().tolist()
            self.indices_with_init_state = []
            for start in self.start_idxes:
                end = start + horizon
                self.indices_with_init_state.append(slice(start, end))
        idxes = np.random.randint(low=0, high=len(self.indices_with_init_state), size=batch_size)
        # (bs, horizon, dim)
        return (
            torch.stack([self.states[self.indices_with_init_state[i]] for i in idxes]),
            torch.stack([self.actions[self.indices_with_init_state[i]] for i in idxes])
        )

class Buffer(SerializedBuffer):

    def __init__(self, buffer_size, state_shape, action_shape, device):
        self._n = 0
        self._p = 0
        self.buffer_size = buffer_size
        self.device = device

        self.states = torch.empty(
            (buffer_size, *state_shape), dtype=torch.float, device=device)
        self.actions = torch.empty(
            (buffer_size, *action_shape), dtype=torch.float, device=device)
        self.rewards = torch.empty(
            (buffer_size, 1), dtype=torch.float, device=device)
        self.dones = torch.empty(
            (buffer_size, 1), dtype=torch.float, device=device)
        self.terminated = torch.empty(
            (buffer_size, 1), dtype=torch.float, device=device)
        self.next_states = torch.empty(
            (buffer_size, *state_shape), dtype=torch.float, device=device)

    def append(self, state, action, reward, done, next_state, terminated=None):
        self.states[self._p].copy_(torch.from_numpy(state))
        self.actions[self._p].copy_(torch.from_numpy(action))
        self.rewards[self._p] = float(reward)
        self.dones[self._p] = float(done)
        if terminated is not None:
            self.terminated[self._p] = float(terminated)
        else:
            self.terminated[self._p] = float(done)
        self.next_states[self._p].copy_(torch.from_numpy(next_state))

        self._p = (self._p + 1) % self.buffer_size
        self._n = min(self._n + 1, self.buffer_size)

    def save(self, path):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        torch.save({
            'state': self.states.clone().cpu(),
            'action': self.actions.clone().cpu(),
            'reward': self.rewards.clone().cpu(),
            'done': self.dones.clone().cpu(),
            'terminated': self.terminated.clone().cpu(),
            'next_state': self.next_states.clone().cpu(),
        }, path)


class RolloutBuffer:

    def __init__(self, buffer_size, state_shape, action_shape, device, with_class=False, mix=1):
        self._n = 0
        self._p = 0
        self.with_class = with_class
        self.mix = mix
        self.buffer_size = buffer_size
        self.total_size = mix * buffer_size
        
        self.states = torch.empty(
            (self.total_size, *state_shape), dtype=torch.float, device=device)
        self.actions = torch.empty(
            (self.total_size, *action_shape), dtype=torch.float, device=device)
        self.rewards = torch.empty(
            (self.total_size, 1), dtype=torch.float, device=device)
        self.dones = torch.empty(
            (self.total_size, 1), dtype=torch.float, device=device)
        self.log_pis = torch.empty(
            (self.total_size, 1), dtype=torch.float, device=device)
        self.next_states = torch.empty(
            (self.total_size, *state_shape), dtype=torch.float, device=device)
        
        if with_class:
            self.class_label = torch.empty(
            (self.total_size, 1), dtype=torch.long, device=device)
            
    def append(self, state, action, reward, done, log_pi, next_state, class_label=None):
        self.states[self._p].copy_(torch.from_numpy(state))
        self.actions[self._p].copy_(torch.from_numpy(action))
        self.rewards[self._p] = float(reward)
        self.dones[self._p] = float(done)
        self.log_pis[self._p] = float(log_pi)
        self.next_states[self._p].copy_(torch.from_numpy(next_state))
        if self.with_class:
            self.class_label[self._p] = class_label

        self._p = (self._p + 1) % self.total_size
        self._n = min(self._n + 1, self.total_size)

    def get(self):
        assert self._p % self.buffer_size == 0
        start = (self._p - self.buffer_size) % self.total_size
        idxes = slice(start, start + self.buffer_size)
        if self.with_class:
            return (
                self.states[idxes],
                self.actions[idxes],
                self.class_label[idxes],
                self.rewards[idxes],
                self.dones[idxes],
                self.log_pis[idxes],
                self.next_states[idxes],
            )
        else:
            return (
                self.states[idxes],
                self.actions[idxes],
                self.rewards[idxes],
                self.dones[idxes],
                self.log_pis[idxes],
                self.next_states[idxes],
            )
            
    def sample(self, batch_size):
        assert self._p % self.buffer_size == 0
        idxes = np.random.randint(low=0, high=self._n, size=batch_size)
        if self.with_class:
            return (
                self.states[idxes],
                self.actions[idxes],
                self.class_label[idxes],
                self.rewards[idxes],
                self.dones[idxes],
                self.log_pis[idxes],
                self.next_states[idxes],
            )
        
        else:
            return (
                self.states[idxes],
                self.actions[idxes],
                self.rewards[idxes],
                self.dones[idxes],
                self.log_pis[idxes],
                self.next_states[idxes],
            )

class RolloutTrajBuffer:

    def __init__(self, buffer_size, state_shape, action_shape, device, mix=1):
        self._n = 0
        self._p = 0
        self.mix = mix
        self.buffer_size = buffer_size
        self.total_size = mix * buffer_size
        # self.max_episode_length = max_episode_length
        self.start_idxes = [0]
        self.path_lengths = []
        self.indices = None

        self.states = torch.empty(
            (self.total_size, *state_shape), dtype=torch.float, device=device)
        self.actions = torch.empty(
            (self.total_size, *action_shape), dtype=torch.float, device=device)
        self.rewards = torch.empty(
            (self.total_size, 1), dtype=torch.float, device=device)
        self.dones = torch.empty(
            (self.total_size, 1), dtype=torch.float, device=device)
        # self.log_pis = torch.empty(
        #     (self.total_size, 1), dtype=torch.float, device=device)
        # self.next_states = torch.empty(
        #     (self.total_size, *state_shape), dtype=torch.float, device=device)

    def append(self, state, action, reward, done):
        self.states[self._p].copy_(torch.from_numpy(state))
        self.actions[self._p].copy_(torch.from_numpy(action))
        self.rewards[self._p] = float(reward)
        self.dones[self._p] = float(done)
        # self.log_pis[self._p] = float(log_pi)
        # self.next_states[self._p].copy_(torch.from_numpy(next_state))

        self._p = (self._p + 1) % self.total_size
        self._n += 1
        if done:
            start_idx = self.start_idxes[-1]
            self.start_idxes.append(self._p)
            self.path_lengths.append(self._p - start_idx)
        # self._n = min(self._n + 1, self.total_size)

    def sample_traj(self, batch_size, horizon):
        assert self._p % self.buffer_size == 0
        if self.indices is None:
            self.indices = self.make_indices(horizon)
        idxes = np.random.randint(low=0, high=len(self.indices), size=batch_size)
        
        # (bs, horizon, dim)
        return (
            torch.stack([self.states[self.indices[i]] for i in idxes]),
            torch.stack([self.actions[self.indices[i]] for i in idxes])
        )

    def make_indices(self, horizon):
        indices = []
        # no padding
        for i, path_length in enumerate(self.path_lengths):
            start_idx = self.start_idxes[i]
            max_start = path_length - horizon + start_idx 
            for start in range(start_idx, max_start):
                end = start + horizon
                indices.append(slice(start, end))
        return indices
    
    def check_path_lengths(self, horizon):
        if len(self.path_lengths) < 1: return False
        if horizon < min(self.path_lengths): return True
        return False
        
    def clear(self):
        self.states = self.states.detach()
        self.states[:, :] = 0
        self.actions[:, :] = 0
        self.rewards[:, :] = 0
        self.dones[:, :] = 0
        # self.log_pis[:, :] = 0
        # self.next_states[:, :] = 0
        self.path_lengths = []
        self.start_idxes = [0]
        self.indices = None
                
if __name__=="__main__":
    buffer = RolloutTrajBuffer(1000, (3,), (2,), 100, device='cpu')
    for epi in range(10):
        for step in range(100):
            state = np.random.rand(3)
            action = np.random.rand(2)
            reward = 1
            done = False
            if step == 99:
                done = True
            buffer.append(state, action, reward, done)
    
    indices = buffer.make_indices(horizon=25)
    import pdb; pdb.set_trace()
    print(len(indices))
    states, actions = buffer.sample(batch_size=256, horizon=25)
    print(states.shape)
    print(actions.shape)
            