import argparse
import functools
import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import warnings
import wandb
from wandb.integration.sb3 import WandbCallback

# cude no visible devices
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
import jax
import jax.numpy as jnp
import rlax
import flax.linen as nn

from stable_baselines3.common import type_aliases
from stable_baselines3.common.callbacks import EvalCallback, CallbackList, BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecMonitor, is_vecenv_wrapped, sync_envs_normalization
from sbx import CrossQ
from sbx.crossq.actor_critic_evaluation_callback import EvalCallback
from sbx.crossq.utils import *

import gymnasium as gym
from shimmy.registration import DM_CONTROL_SUITE_ENVS


os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['WANDB_DIR'] = '/tmp'

parser = argparse.ArgumentParser()
parser.add_argument("-env",         type=str, required=False, default="Walker2d-v4", help="Set Environment.")
parser.add_argument("-seed",        type=int, required=False, default=1, help="Set Seed.")
parser.add_argument("-log_freq",    type=int, required=False, default=300, help="how many times to log during training")

parser.add_argument('-wandb_project', type=str, required=False, default='sbx_feat', help='wandb project name')
parser.add_argument("-wandb_mode",    type=str, required=False, default='disabled', choices=['disabled', 'online'], help="enable/disable wandb logging")

parser.add_argument("-total_timesteps",   type=int,   required=False, default=5e6, help="total number of training steps")

experiment_time = time.time()
args = parser.parse_args()

seed = args.seed
total_timesteps = int(args.total_timesteps)
eval_freq = max(5_000_000 // args.log_freq, 1)

group = f'CrossQ_{args.env}_b1=.5_2'

args_dict = vars(args)

with wandb.init(
    entity='ias', # TODO: remove for publication
    project=args.wandb_project,
    name=f"seed={seed}",
    group=group,
    tags=[],
    sync_tensorboard=True,
    config=args_dict,
    settings=wandb.Settings(start_method="fork") if is_slurm_job() else None,
    mode=args.wandb_mode
) as wandb_run:
    
    # SLURM maintainance
    if is_slurm_job():
        print(f"SLURM_JOB_ID: {os.environ.get('SLURM_JOB_ID')}")
        wandb_run.summary['SLURM_JOB_ID'] = os.environ.get('SLURM_JOB_ID')

    training_env = gym.make(args.env)

    model = CrossQ(
        "MlpPolicy",
        training_env,
        # policy_kwargs=dict({
        #     'activation_fn': nn.relu,
        #     'layer_norm': layer_norm,
        #     'batch_norm': bool(args.bn),
        #     'batch_norm_momentum': float(args.bn_momentum),
        #     'batch_norm_mode': args.bn_mode,
        #     'ofn': args.ofn,
        #     'dropout_rate': dropout_rate,
        #     'n_critics': args.n_critics,
        #     'net_arch': net_arch,
        #     'optimizer_class': optax.adam,
        #     'optimizer_kwargs': dict({
        #         'b1': args.adam_b1,
        #         'b2': 0.999 # default
        #     })
        # }),
        # policy_delay=args.policy_delay,
        # crossq_style=bool(args.crossq_style),
        # td3_mode=td3_mode,
        # use_bnstats_from_live_net=bool(args.bnstats_live_net),
        # policy_q_reduce_fn=policy_q_reduce_fn,
        # learning_starts=5000,
        # learning_rate=args.lr,
        # qf_learning_rate=args.lr,
        # tau=args.tau,
        # gamma=0.99 if not args.env == 'Swimmer-v4' else 0.9999,
        # verbose=0,
        # buffer_size=1_000_000,
        # seed=seed,
        # stats_window_size=1,  # don't smooth the episode return stats over time
        tensorboard_log=f"logs/{group + 'seed=' + str(seed) + '_time=' + str(experiment_time)}/",
    )

    # Create log dir where evaluation results will be saved
    eval_log_dir = f"./eval_logs/{group + 'seed=' + str(seed) + '_time=' + str(experiment_time)}/eval/"
    qbias_log_dir = f"./eval_logs/{group + 'seed=' + str(seed) + '_time=' + str(experiment_time)}/qbias/"
    os.makedirs(eval_log_dir, exist_ok=True)
    os.makedirs(qbias_log_dir, exist_ok=True)

    # Create callback that evaluates agent
    eval_callback = EvalCallback(
        make_vec_env(args.env, n_envs=1, seed=seed),
        jax_random_key_for_seeds=args.seed,
        best_model_save_path=None,
        log_path=eval_log_dir, eval_freq=eval_freq,
        n_eval_episodes=1, deterministic=True, render=False
    )

    callback_list = CallbackList([eval_callback, WandbCallback(verbose=0,)])
    model.learn(total_timesteps=total_timesteps, progress_bar=True, callback=callback_list)
