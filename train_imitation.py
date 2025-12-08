import wandb
import os
import argparse, json, random, string
from datetime import datetime
import torch

from imitation.env import make_env
from imitation.buffer import SerializedBuffer
from imitation.algo import ALGOS
from imitation.imit_trainer import Trainer 


def run(args):
    time = datetime.now().strftime("%Y%m%d-%H%M")
    random_string = ''.join(random.choice(string.ascii_lowercase) for i in range(5))
    trajs_size = args.buffer.split('/')[-1]
    log_dir = os.path.join(
        './output', args.env_id, args.algo, 
        f'{args.num_modes}', trajs_size,  
        f'{time}-seed{args.seed}-{random_string}')
    os.makedirs(log_dir, exist_ok=True)
    print(f"log dir {log_dir}")
    # save arguments as .txt    
    with open(log_dir+'/commandline_args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    wandb.init(project=f'{args.env_id}', config=args)
    wandb.config['string'] = random_string
    
    env = make_env(args.env_id, args.mode_idx)
    env_test = make_env(args.env_id, mode_idx=None)
    buffer_exp = SerializedBuffer(
        directory_path=args.buffer, 
        device=torch.device("cuda" if args.cuda else "cpu"),
        num_modes=args.num_modes
    )

    algo = ALGOS[args.algo](
        buffer_exp=buffer_exp,
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        rollout_length=args.rollout_length,
        horizon=args.horizon,
        act_seq_length=args.act_seq_length,
        n_timesteps=args.n_timesteps,
        n_pi_epochs=args.n_pi_epochs,
        batch_size=args.batch_size,
        use_obs_norm=args.use_obs_norm,
        coef=args.coef,
        device=torch.device("cuda" if args.cuda else "cpu"),
        seed=args.seed,
    )
    trainer = Trainer(
        env_id=args.env_id,
        mode_list=buffer_exp.mode_list,
        env=env,
        env_test=env_test,
        algo=algo,
        algo_id=args.algo,
        log_dir=log_dir,
        one_step_control=args.one_step_control,
        num_steps=args.num_steps,
        eval_interval=args.eval_interval,
        seed=args.seed
    )
    trainer.train()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--algo', type=str, default='dpail')
    p.add_argument('--env_id', type=str, default='Ant-v3')
    p.add_argument('--mode_idx', type=int, default=0)
    p.add_argument('--num_modes', type=int, default=4)
    p.add_argument('--buffer', type=str, required=True)
    p.add_argument('--rollout_length', type=int, default=5000)
    p.add_argument('--num_steps', type=int, default=5000000)
    p.add_argument('--batch_size', type=int, default=1000)
    p.add_argument('--eval_interval', type=int, default=10000)
    #---- sequence training (diffusion, d3o, asaf)
    p.add_argument('--horizon', type=int, default=8)
    #---- sequence control (diffusion, d3o)
    p.add_argument('--act_seq_length', type=int, default=4)
    #---- diffusion model (diffusion, d3o, diffail, drail)
    p.add_argument('--n_timesteps', type=int, default=50)
    #---- n_pi_epochs (diffusion, d3o, asaf)
    p.add_argument('--n_pi_epochs', type=int, default=1000)
    #---- coef (d3o, infogail)
    p.add_argument('--coef', type=float, default=2)
    #----
    p.add_argument('--one_step_control', action='store_true')
    p.add_argument('--use_obs_norm', action='store_true')
    p.add_argument('--cuda', action='store_true')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    run(args)
