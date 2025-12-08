import torch
from torch import nn
import torch.nn.functional as F

from .utils import build_mlp
DISC_LOGIT_INIT_SCALE = 1.0

class GAILDiscrim(nn.Module):

    def __init__(self, state_shape, action_shape, hidden_units=(100, 100),
                 hidden_activation=nn.Tanh()):
        super().__init__()

        self.net = build_mlp(
            input_dim=state_shape[0] + action_shape[0],
            output_dim=1,
            hidden_units=hidden_units,
            hidden_activation=hidden_activation
        )

    def forward(self, states, actions):
        return self.net(torch.cat([states, actions], dim=-1))

    def calculate_reward(self, states, actions):
        # PPO(GAIL) is to maximize E_{\pi} [-log(1 - D)].
        with torch.no_grad():
            return -F.logsigmoid(-self.forward(states, actions))


class InfoGAILDiscrim(nn.Module):

    def __init__(self, state_shape, action_shape, dim_c=None, hidden_units=(100, 100),
                 hidden_activation=nn.Tanh()):
        super().__init__()

        self.d_net = build_mlp(
            input_dim=state_shape[0] + action_shape[0],
            output_dim=1,
            hidden_units=hidden_units,
            hidden_activation=hidden_activation
        )
        
        self.c_net =  build_mlp(
            input_dim=state_shape[0] + action_shape[0],
            output_dim=dim_c,
            hidden_units=hidden_units,
            hidden_activation=hidden_activation
        )
        
    def forward(self, states, actions):
        x = torch.cat([states, actions], dim=-1)
        d_logit = self.d_net(x)
        c = torch.softmax(self.c_net(x), -1)
        return d_logit, torch.clamp(c, min=1e-20)

    # PPO(InfoGAIL) is to maximize E_{\pi} [-log(1 - D) + \coef log Q(c| s, a)].
    def calculate_reward(self, states, actions, class_label_gt, coef=0.1):
        with torch.no_grad():
            logit, pred_c = self.forward(states, actions)
            log_prob_c = torch.log(pred_c[torch.arange(len(class_label_gt)), class_label_gt.view(-1)])
        return -F.logsigmoid(-logit) + coef * log_prob_c.unsqueeze(-1)

class AIRLDiscrim(nn.Module):

    def __init__(self, state_shape, gamma,
                 hidden_units_r=(64, 64),
                 hidden_units_v=(64, 64),
                 hidden_activation_r=nn.ReLU(inplace=True),
                 hidden_activation_v=nn.ReLU(inplace=True)):
        super().__init__()

        self.g = build_mlp(
            input_dim=state_shape[0],
            output_dim=1,
            hidden_units=hidden_units_r,
            hidden_activation=hidden_activation_r
        )
        self.h = build_mlp(
            input_dim=state_shape[0],
            output_dim=1,
            hidden_units=hidden_units_v,
            hidden_activation=hidden_activation_v
        )

        self.gamma = gamma

    def f(self, states, dones, next_states):
        rs = self.g(states)
        vs = self.h(states)
        next_vs = self.h(next_states)
        return rs + self.gamma * (1 - dones) * next_vs - vs

    def forward(self, states, dones, log_pis, next_states):
        # Discriminator's output is sigmoid(f - log_pi).
        return self.f(states, dones, next_states) - log_pis

    def calculate_reward(self, states, dones, log_pis, next_states):
        with torch.no_grad():
            logits = self.forward(states, dones, log_pis, next_states)
            return -F.logsigmoid(-logits)
