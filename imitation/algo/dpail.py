import wandb
import numpy as np
import os
import torch, copy
from torch import nn
from torch.optim import Adam
from torch.nn import functional as F
from .base import Algorithm
from imitation.buffer import RolloutTrajBuffer
from imitation.network import StateIndependentPolicy, TemporalUnetTransformer, GaussianDiffusion, InvDyancmis
from imitation.utils.utils import EMA, LimitsNormalizer
from imitation.utils.arrays import batch_to_device, to_np, to_torch, to_device, apply_dict

class DPAIL(Algorithm):

    def __init__(self,
                 buffer_exp,
                 state_shape, 
                 action_shape,
                 rollout_length=2048, 
                 batch_size=256,
                 horizon=8,
                 act_seq_length=8, 
                 n_timesteps=20,
                 use_obs_norm=True,
                 lr_actor=2e-4, 
                 units_actor=(1, 2, 4, 8),
                 state_sequence=False,
                 inv_dynamics=True,
                 n_pi_epochs=100, 
                 coef=2.0,
                 max_grad_norm=0.5, 
                 ema_decay=0.1,
                 device='cuda',
                 seed=0,
                 **kwargs):
        super().__init__(state_shape, action_shape, device, seed, gamma=None)
        
        # Expert's buffer
        self.buffer_exp = buffer_exp

        # Rollout buffer
        self.buffer = RolloutTrajBuffer(
            buffer_size=rollout_length,
            state_shape=state_shape,
            action_shape=action_shape,
            device=device,
            # mix=mix_buffer
        )
        # Actor
        params, self.state_sequence, self.inv_dynacmis = [], False, False
        # s sequence
        if state_sequence:
            self.state_sequence = True
            action_dim = 0
            model = TemporalUnetTransformer(horizon, transition_dim=state_shape[0], 
                                            cond_dim=0, conditional=False,
                                            dim_mults=units_actor, condition_dropout=0.25, calc_energy=False)
        # (s, a) sequence
        else:
            action_dim = action_shape[0]
            model = TemporalUnetTransformer(horizon, transition_dim=state_shape[0]+action_shape[0], 
                                            cond_dim=0, conditional=False,
                                            dim_mults=units_actor, condition_dropout=0.25, calc_energy=False)
        self.model = model.to(device)
        self.actor = GaussianDiffusion(self.model, state_shape[0], action_dim, horizon, n_timesteps=n_timesteps, device=device)
        self.exp_actor = copy.deepcopy(self.actor)
        params.append({'params': self.exp_actor.parameters(), 'name': 'exp_actor'})
                    
        self.optim_actor = Adam(params, lr=lr_actor)
        self.rollout_length = rollout_length
        self.batch_size = batch_size
        self.horizon = horizon
        self.act_seq_length = act_seq_length
        self.n_timesteps = n_timesteps
        self.n_pi_epochs = n_pi_epochs
        self.coef = coef
        self.max_grad_norm = max_grad_norm
        
        self.ema = EMA(ema_decay) # ema_decay * old + (1-ema_decay) * new_one
        self.learning_steps = 0
        self.learning_steps_d3o = 0
        self.eval_act_seq_length = 4
        
        # Observation normalizer
        self.normalizer = None
        if use_obs_norm:
            self.normalizer = LimitsNormalizer(to_np(self.buffer_exp.states))
        self.actor.eval()

    def is_update(self, step):
        # if step < 0: return True
        return step % self.rollout_length == 0

    @torch.no_grad()
    def explore_action_sequence(self, state):
        if self.normalizer is not None:
            state = self.normalizer.normalize(state)
        state = torch.tensor(state, dtype=torch.float, device=self.device).reshape(-1,self.state_shape[0])
        action_sequence = self.actor(state=state)[:, : , self.state_shape[0]:] # (bs, seq_len, dim)
        return to_np(action_sequence)[0]
            
    def step(self, env, state, t, step):
        if (t % self.act_seq_length == 0) or self.is_update(step - 1) : 
            self.action_sequence = self.explore_action_sequence(state)
        action = self.action_sequence[t % self.act_seq_length]
        next_state, reward, done, _ = env.step(action)
        self.buffer.append(state, action, reward, done)
        t += 1
        if done:
            t = 0
            next_state = env.reset()
        return next_state, t

    def update(self):
        self.learning_steps += 1
        # (bs, horizon, dim)
        gen_check = False
        if self.buffer.check_path_lengths(horizon=self.horizon):
            gen_check = True
            if self.normalizer is not None:
                self.normalizer.update_normalizer(to_np(self.buffer.states))
            
        for _ in range(self.n_pi_epochs):
            if gen_check:
                self.gen_coef = 1.0
                pi_states, pi_actions = self.buffer.sample_traj(batch_size=self.batch_size, horizon=self.horizon)
            else:
                self.gen_coef = 0
                pi_states, pi_actions = self.buffer_exp.sample_traj(batch_size=self.batch_size, horizon=self.horizon)
            exp_states, exp_actions = self.buffer_exp.sample_traj(batch_size=self.batch_size, horizon=self.horizon)
            
            self.update_actor(exp_states, exp_actions, pi_states, pi_actions)
        self.ema.update_model_average(self.actor, self.exp_actor)
        self.buffer.clear()
        
    def update_actor(self, exp_s, exp_a, gen_s, gen_a):
        # (bs, horizon, dim)
        self.learning_steps_d3o += 1
        trajs = torch.cat([exp_s, gen_s], dim=0)
        
        if self.normalizer is not None:
            exp_s = self.normalizer.normalize_from_torch(exp_s, self.device)
            gen_s = self.normalizer.normalize_from_torch(gen_s, self.device)
            
        if not self.state_sequence:
            cat_trajs = torch.cat([torch.cat([exp_s, exp_a], dim=-1), 
                                   torch.cat([gen_s, gen_a], dim=-1)], dim=0)
        else: cat_trajs = torch.cat([exp_s, gen_s], dim=0)
        
        t = torch.randint(0, self.n_timesteps, (self.batch_size*2,), device=self.device).long() # (bs,)
        target_mse_exp, target_mse_gen = self.exp_actor.mse_loss(cat_trajs, t).chunk(2, dim=0)
        with torch.no_grad():
            prev_mse_exp, prev_mse_gen = self.actor.mse_loss(cat_trajs, t).chunk(2, dim=0)
        # E_exp
        exp_inside_term = (prev_mse_exp - target_mse_exp).sum(-1)
        loss_exp = -F.logsigmoid(self.coef * self.n_timesteps * exp_inside_term).mean()
        # E_g
        # gen_inside_term = (target_mse_gen - prev_mse_gen).sum(-1) # for test
        gen_inside_term = (target_mse_gen - 0.).sum(-1)
        loss_gen = -F.logsigmoid(self.coef * self.n_timesteps * gen_inside_term).mean()
        loss_actor = loss_exp + self.gen_coef * loss_gen
        self.optim_actor.zero_grad()
        (loss_actor).backward()
        nn.utils.clip_grad_norm_(self.exp_actor.parameters(), self.max_grad_norm)
        self.optim_actor.step()

        if self.learning_steps_d3o % self.n_pi_epochs == 0:
            print(f'{self.learning_steps_d3o} | loss/actor {round(loss_actor.item(), 5)}')
            wandb.log({
                'loss/actor': loss_actor.item(),
                'loss/gen_coef': self.gen_coef,
                'loss/loss_exp': loss_exp.item(),
                'loss/loss_gen': loss_gen.item(),
                'loss/target_mse_exp': target_mse_exp.mean().item(),
                'loss/target_mse_gen': prev_mse_gen.mean().item(),
                'loss/prev_mse_exp': prev_mse_exp.mean().item(),
                'loss/prev_mse_gen': prev_mse_gen.mean().item(),
                'loss/exp_inside_term': exp_inside_term.mean().item(),
                'loss/gen_inside_term': gen_inside_term.mean().item(),
                })

    def save_models(self, save_dir):
        super().save_models(save_dir)
        # We only save actor to reduce workloads.
        torch.save(
            {'actor':self.actor.state_dict(), 
             'mins': self.normalizer.mins,  # np.array
             'maxs': self.normalizer.maxs },
            os.path.join(save_dir, 'actor.pth')
        )

    def load_models(self, save_dir):
        data = torch.load(os.path.join(save_dir, 'actor.pth'))
        self.actor.load_state_dict(data['actor'])
        self.actor.to(self.device)
        self.normalizer.mins = data['mins']
        self.normalizer.maxs = data['maxs']
        print("Load model weight")    
