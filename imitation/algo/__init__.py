from .ppo import PPO
from .gail import GAIL
from .dpail import DPAIL
ALGOS = {
    'gail': GAIL,
    'dpail': DPAIL,    
}
