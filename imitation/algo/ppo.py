import wandb
import torch, os
from torch import nn
from torch.optim import Adam

from .base import Algorithm
from imitation.buffer import RolloutBuffer
from imitation.network import StateIndependentPolicy, StateFunction

@torch.no_grad()
def calculate_gae(values, rewards, dones, next_values, gamma, lambd):
    # Calculate TD errors.
    deltas = rewards + gamma * next_values * (1 - dones) - values
    # Initialize gae.
    gaes = torch.empty_like(rewards)
    # Calculate gae recursively from behind.
    gaes[-1] = deltas[-1]
    for t in reversed(range(rewards.size(0) - 1)):
        gaes[t] = deltas[t] + gamma * lambd * (1 - dones[t]) * gaes[t + 1]

    return gaes + values, (gaes - gaes.mean()) / (gaes.std() + 1e-8)


class PPO(Algorithm):

    def __init__(self, state_shape, action_shape, device, seed, gamma=0.995,
                 rollout_length=2048, mix_buffer=20, lr_actor=3e-4,
                 lr_critic=3e-4, units_actor=(64, 64), units_critic=(64, 64),
                 epoch_ppo=10, clip_eps=0.2, lambd=0.97, coef_ent=0.0,
                 max_grad_norm=10.0, with_class=False):
        super().__init__(state_shape, action_shape, device, seed, gamma)

        # Rollout buffer.
        self.buffer = RolloutBuffer(
            buffer_size=rollout_length,
            state_shape=state_shape,
            action_shape=action_shape,
            device=device,
            mix=mix_buffer,
            with_class=with_class
        )

        # Actor.
        self.actor = StateIndependentPolicy(
            state_shape=state_shape,
            action_shape=action_shape,
            hidden_units=units_actor,
            hidden_activation=nn.Tanh()
        ).to(device)

        # Critic.
        self.critic = StateFunction(
            state_shape=state_shape,
            hidden_units=units_critic,
            hidden_activation=nn.Tanh()
        ).to(device)

        self.optim_actor = Adam(self.actor.parameters(), lr=lr_actor)
        self.optim_critic = Adam(self.critic.parameters(), lr=lr_critic)

        self.learning_steps_ppo = 0
        self.rollout_length = rollout_length
        self.epoch_ppo = epoch_ppo
        self.clip_eps = clip_eps
        self.lambd = lambd
        self.coef_ent = coef_ent
        self.max_grad_norm = max_grad_norm
        self.device = device

    def is_update(self, step):
        return step % self.rollout_length == 0

    def step(self, env, state, t, step):
        t += 1
        action, log_pi = self.explore(state)
        next_state, reward, done, _ = env.step(action)
        mask = False if t == env._max_episode_steps else done
        self.buffer.append(state, action, reward, mask, log_pi, next_state)
        if done:
            t = 0
            next_state = env.reset()

        return next_state, t

    def update(self, writer=None):
        self.learning_steps += 1
        states, actions, rewards, dones, log_pis, next_states = \
            self.buffer.get()
        self.update_ppo(
            states, actions, rewards, dones, log_pis, next_states)

    def update_ppo(self, states, actions, rewards, dones, log_pis, next_states):
        with torch.no_grad():
            values = self.critic(states)
            next_values = self.critic(next_states)
        # import pdb; pdb.set_trace()
        with torch.no_grad():
            targets, gaes = calculate_gae(
                values, rewards, dones, next_values, self.gamma, self.lambd)
        
        for _ in range(self.epoch_ppo):
            self.learning_steps_ppo += 1
            self.update_critic(states, targets)
            self.update_actor(states, actions, log_pis, gaes)

    def update_critic(self, states, targets):
        loss_critic = (self.critic(states) - targets).pow_(2).mean()
        
        self.optim_critic.zero_grad()
        loss_critic.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.optim_critic.step()

        if self.learning_steps_ppo % self.epoch_ppo == 0:
            print(f'{self.learning_steps} | loss/critic {round(loss_critic.item(), 3)} |')
            wandb.log({
                'PI/critic_loss': round(loss_critic.item(), 3)
            })

    def update_actor(self, states, actions, log_pis_old, gaes):
        log_pis = self.actor.evaluate_log_pi(states, actions)
        entropy = -log_pis.mean()

        ratios = (log_pis - log_pis_old).exp_()
        loss_actor1 = -ratios * gaes
        loss_actor2 = -torch.clamp(
            ratios,
            1.0 - self.clip_eps,
            1.0 + self.clip_eps
        ) * gaes
        loss_actor = torch.max(loss_actor1, loss_actor2).mean()

        self.optim_actor.zero_grad()
        # (loss_actor - self.coef_ent * entropy).backward(retain_graph=False)
        (loss_actor - self.coef_ent * entropy).backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.optim_actor.step()

        if self.learning_steps_ppo % self.epoch_ppo == 0:
            print(f'{self.learning_steps_ppo} | loss/actor {round(loss_actor.item(), 3)} |')
            wandb.log({
                'PI/actor_loss': loss_actor.item(),
                'PI/actor_entropy': entropy.item()
            })

    def save_models(self, save_dir):
        super().save_models(save_dir)
        # We only save actor to reduce workloads.
        torch.save(
            self.actor.state_dict(),
            os.path.join(save_dir, 'actor.pth')
        )
    def load_models(self, save_dir):
        self.actor.load_state_dict(torch.load(os.path.join(save_dir, 'actor.pth')))
        self.actor.to(self.device)
        print("Load model weight")    
        
        