import numpy as np
import torch
from torch import nn
import copy
import einops
from imitation.utils.helpers import cosine_beta_schedule, linear_beta_schedule, vp_beta_schedule, \
                            extract, apply_conditioning, Losses


class GaussianDiffusion(nn.Module):
    def __init__(self, model, observation_dim, action_dim, horizon,
                 n_timesteps=100, loss_type='l2', clip_denoised=True, predict_epsilon=True,
                 loss_discount=1.0, loss_weights=None, conditional=False, 
                 action_weight=1.,
                 beta_schedule='cosine', 
                 device='cpu'):
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.obsact_dim = observation_dim + action_dim
        self.transition_dim = observation_dim + action_dim
        self.horizon = horizon
        
        self.model = model
        self.conditional = conditional
        self.device = device
        
        if beta_schedule == 'linear':
            betas = linear_beta_schedule(n_timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(n_timesteps)
        elif beta_schedule == 'vp':
            betas = vp_beta_schedule(n_timesteps)
        else:
            NotImplementedError(beta_schedule)
        
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])
        
        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon
        
        self.register_buffer('betas', betas.to(device))
        self.register_buffer('alphas_cumprod', alphas_cumprod.to(device))
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev.to(device))
        
        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod).to(device))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod).to(device))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod).to(device))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod).to(device))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod -1).to(device))
        
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance.to(device))
        
        ## log calculation clipped because the posterior variance 
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', torch.log(torch.clamp(posterior_variance, min=1e-20)).to(device))
        self.register_buffer('posterior_mean_coef1', (betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)).to(device))
        self.register_buffer('posterior_mean_coef2', ((1. -  alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)).to(device))
        
        ## get loss coefficients and initialize objective
        loss_weights = self.get_loss_weights(action_weight, loss_discount, loss_weights)
        self.loss_fn = Losses[loss_type](loss_weights, self.action_dim)
        self.log_sigam_fn = nn.LogSigmoid()

    def get_loss_weights(self, action_weight, discount, weights_dict):
        '''
            sets loss coefficients for trajectory
            action_weight   : float
                coefficient on first action loss
            discount   : float
                multiplies t^th timestep of trajectory loss by discount**t
            weights_dict    : dict
                { i: c } multiplies dimension i of observation loss by c
        '''
        self.action_weight = torch.tensor(action_weight, dtype=torch.float32)
        dim_weights = torch.ones(self.transition_dim, dtype=torch.float32)
        
        ## set loss coefficients for dimensions of observation
        if weights_dict is None: weights_dict = {}
        for ind, w in weights_dict.items():
            dim_weights[self.action_dim + ind] *= w
        
        # decay loss with trajectory timestep: discount**t
        discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        loss_weights = torch.einsum('h,t->ht', discounts, dim_weights)
        loss_weights = dim_weights * discount
        
        ## manually set a0 weight
        loss_weights[:self.action_dim] = self.action_weight
        return loss_weights.to(self.device)
    
    #### not used
    @torch.no_grad()
    def score(self, x, t, cond, goal):
        epsilon = self.model(x, t, cond, goal)
        return - extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)**(-1) * epsilon
    
    def predict_start_from_noise(self, x_t, t, noise):
        '''
            q(x_t | x_0) --> x_0 = a * x_t - b * epsilon
        '''
        if self.predict_epsilon:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise
        
    def q_posterior(self, x_start, x_t, t):
        '''
            q(x_{t-1} | x_t, x_0) ~ N( mu_tilde(x_t, x_0), beta_tilde )
        '''
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, cond):
        '''
            p_theta(x_{t-1} | x_t)
        '''
        if self.model.calc_energy:
            assert self.predict_epsilon
            x = torch.tensor(x, requires_grad=True)
            t = torch.tensor(t, dtype=torch.float, requires_grad=True)
        
        epsilon = self.model(x, t, cond)                        
        t = t.detach().to(torch.int64)
        x_recon = self.predict_start_from_noise(x, t=t, noise=epsilon)
        
        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)
        else:
            assert RuntimeError()
        
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
   
        return model_mean, posterior_variance, posterior_log_variance
    
    @torch.no_grad()
    def p_sample(self, x, t, cond):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, cond=cond)
        noise = 0.5 * torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t==0).float()).reshape(b, *((1,) * (len(x.shape)-1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def n_step_guided_p_sample(self, x, t, cond, guide, scale=0.1, t_stopgrad=0, n_guide_steps=1, scale_grad_by_std=True):
        model_log_variance = extract(self.posterior_log_variance_clipped, t, x.shape)
        model_std = torch.exp(0.5 * model_log_variance)
        model_var = torch.exp(model_log_variance)

        for _ in range(n_guide_steps):
            with torch.enable_grad():
                y, grad = guide.gradients(x, cond) # x : trajectoreis, cond: expert batch
            if scale_grad_by_std:
                grad = model_var * grad

            grad[t < t_stopgrad] = 0

            x = x + scale * grad
            # x = apply_conditioning(x, cond, self.action_dim)

        model_mean, _, model_log_variance = self.p_mean_variance(x=x, cond=cond, t=t)

        # no noise when t == 0
        noise = torch.randn_like(x)
        noise[t == 0] = 0

        return model_mean + model_std * noise
    
    @torch.no_grad()
    def p_sample_loop(self, shape, state, cond, guided_sample=False, guide=None, return_diffusion=False):
    
        batch_size = shape[0]
        x = 0.5 * torch.randn(shape, device=self.device)
        # apply conditioning
        if self.observation_dim > 0:
            x[:, 0, :self.observation_dim] = state.clone()
        
        if return_diffusion: diffusion = [x]
        
        # progress = utils.Progress(self.n_timesteps)
        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            if guided_sample:
                assert guide is not None
                x = self.n_step_guided_p_sample(x, timesteps, cond, guide)
            else:
                x = self.p_sample(x, timesteps, cond)
            # apply conditioning
            x[:, 0, :self.observation_dim] = state.clone()
    
            # progress.update({'t': i})
            if return_diffusion: diffusion.append(x)
        # progress.close()
        
        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x
        
    @torch.no_grad()
    def p_sample_loop_(self, shape, states, cond, guided_sample=False, guide=None, return_diffusion=False):
        len_state = len(states)
        batch_size = shape[0]
        x = 0.5 * torch.randn(shape, device=self.device)
        # apply conditioning
        for i in range(len_state):
            x[:, i, :self.observation_dim] = states[i].clone()
        
        if return_diffusion: diffusion = [x]
        
        # progress = utils.Progress(self.n_timesteps)
        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            if guided_sample:
                assert guide is not None
                x = self.n_step_guided_p_sample(x, timesteps, cond, guide)
            else:
                x = self.p_sample(x, timesteps, cond)
            # apply conditioning
            for i in range(len_state):
                x[:, i, :self.observation_dim] = states[i].clone()
    
            # progress.update({'t': i})
            if return_diffusion: diffusion.append(x)
        # progress.close()
        
        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x
        
    @torch.no_grad()
    def conditional_sample(self, state, cond=None, guided_sample=False, *args, **kwargs):
        batch_size = state.shape[0]
        shape =  (batch_size, self.horizon, self.transition_dim)
        return self.p_sample_loop(shape, state, cond, guided_sample, *args, **kwargs)
    
    @torch.no_grad()
    def conditional_sample_history(self, state, cond=None, guided_sample=False, *args, **kwargs):
        shape =  (1, self.horizon, self.transition_dim)
        if len(state) == 1:
            return self.p_sample_loop(shape, state, cond, guided_sample, *args, **kwargs)
        else:
            return self.p_sample_loop_(shape, state, cond, guided_sample, *args, **kwargs)
    
    #------------------------------------------ training ------------------------------------------#
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
        return sample
    
    def p_losses(self, x_start, t, state=None, cond=None, with_predict_epsilon=False): #  goal, has_object=False):
        noise = torch.randn_like(x_start) # (bs, seq_len, dim)
        x_start = x_start.float()
        x_noisy = self.q_sample(x_start, t, noise) # (bs, seq_len, dim)
        # apply conditioning
        if state is not None:
            x_noisy[:, 0, :self.observation_dim] = state.clone()
                
        if self.model.calc_energy:
            assert self.predict_epsilon
            x_noisy.requires_grad = True
            t = torch.tensor(t, dtype=torch.float, requires_grad=True)
            # goal.requires_grad = True
            noise.requires_grad = True
        
        x_recon = self.model(x_noisy, t, y=cond)
        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)
        
        if not self.predict_epsilon:
            # apply conditioning
            if state is not None:
                x_recon[:, 0, :self.observation_dim] = state.clone()        
        assert noise.shape == x_recon.shape
        
        if self.predict_epsilon:
            loss = self.loss_fn(x_recon, noise)
        else:
            loss = self.loss_fn(x_recon, x_start)
        
        if with_predict_epsilon: return loss, x_recon
        
        return loss
    
    def loss(self, trajectories, cond=None): 
        batch_size = len(trajectories)
        
        state = trajectories[:, 0, :self.observation_dim] # (bs, state_dim)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=trajectories.device).long() # (bs)
        loss = self.p_losses(trajectories, t, state, cond=cond) # cond, goal, has_object)
    
        return loss.mean()

    def mse_loss(self, trajectories, t=None, cond=None, with_predict_epsilon=False, state_condition=True): 
        batch_size = len(trajectories)
        state = None
        if state_condition:
            state = trajectories[:, 0, :self.observation_dim] # (bs, state_dim)
        if t is None:
            t = torch.randint(0, self.n_timesteps, (batch_size,), device=trajectories.device).long() # (bs)
        output = self.p_losses(trajectories, t, state, cond=cond, with_predict_epsilon=with_predict_epsilon)
        return output
    
    def predict_noise(self, trajectories, t=None, cond=None):
        batch_size = len(trajectories)
        state = trajectories[:, 0, :self.observation_dim] # (bs, state_dim)
        if t is None:
            t = torch.randint(0, self.n_timesteps, (batch_size,), device=trajectories.device).long() # (bs)
        
        noise = torch.randn_like(trajectories) # (bs, seq_len, dim)
        x_start = x_start.float()
        x_noisy = self.q_sample(trajectories, t, noise) # (bs, seq_len, dim)
        # apply conditioning
        x_noisy[:, 0, :self.observation_dim] = state.clone()
        x_recon = self.model(x_noisy, t, y=cond)
        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)
        assert noise.shape == x_recon.shape
        return x_recon
    
    def forward(self, state, cond=None, *args, **kwargs): # goal, *args, **kwargs):
        return self.conditional_sample(state, cond, *args, **kwargs)

