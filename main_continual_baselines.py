import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import gym
import random
import cv2
from procgen import ProcgenEnv
import os
from tensorboardX import SummaryWriter
import sys

import warnings
warnings.filterwarnings(action='ignore')

from arguments import get_args_continual_baselines
from utils import *
from models import *
from ppo import *

args = get_args_continual_baselines()

'''cuda setting'''

device = 'cuda' if torch.cuda.is_available else 'cpu'
if device != 'cuda': print("No cuda detected. Consider colab.research.google.com for free GPUs.")

'''create agent (see models)'''

agent = ImpalaCNN().to(device)
get_n_params(agent)

'''make envs'''

n_envs = args.num_envs
n_levels = 1
env_name = args.env_name
start_levels = [i for i in range(args.start_level_min, args.start_level_max+1)]

envs = []

for start_level in start_levels:
    env_tmp = ProcgenEnv(num_envs=n_envs, 
                     env_name=env_name,
                     start_level=start_level,
                     num_levels=n_levels, 
                     distribution_mode='easy')

    env_tmp = VecExtractDictObs(env_tmp, "rgb")
    env_tmp = VecMonitor(env_tmp)
    env_tmp = VecNormalize(env_tmp, ob=False)
    envs.append(env_tmp)
    
env = envs[0]

'''Hyperparameters all taken from procgen paper'''

lr = 5e-4
n_steps = 256
opt = torch.optim.Adam(agent.parameters(), lr=lr, eps=1e-5)
n_obs_per_round = n_envs * n_steps;
target_n_obs = 10_000_000 * len(envs)
n_minibatches = 8
bs = n_obs_per_round // n_minibatches
n_rounds = target_n_obs // n_obs_per_round
env_change_interval = n_rounds // len(start_levels)

accumulation_steps = 1

entropy_coef = .01
value_loss_coef = .5
n_opt_epochs = 3

n_obs_per_round, n_rounds, bs

verbose = False
eval_steps = n_steps

'''dir settings'''
filename = sys.argv[0]
filename = filename[:filename.index(".")]

args.save_dir = os.path.join(args.save_dir, filename)
args.log_dir = os.path.join(args.log_dir, filename)

id_string = env_name + "_stlv" + str(start_levels[0]) + "~" + str(start_levels[-1]) + "_lvnum" + str(n_levels)
save_path = os.path.join(args.save_dir, id_string +".pt")
log_path = os.path.join(args.log_dir, id_string) + "/"

try:
    os.makedirs(args.save_dir)
except OSError:
    pass

'''tensorboard settings'''
writer = SummaryWriter(log_path)

'''print settings'''

print()
print("================================")
print("use device :", device)
print("env :", env_name)
print("seed num :", len(envs))
print("n_rounds :", n_rounds)
print("save_path :",save_path)
print("log_path :",log_path)
print("target_n_obs :", target_n_obs)
print("================================")
print()

'''main'''

scores = []; losses = []; total_steps = 0; env_index = 0

last_frame_ = env.reset()
for i in range(1, n_rounds):
    
    '''get trajectories for n_steps'''
    
    with torch.no_grad():
        agent.eval()
        if i%env_change_interval==0:
            if len(envs)-1 > env_index:
                env_index += 1
            env = envs[env_index]
            last_frame_ = env.reset()
        
        frame = last_frame_
        
        frames_, rewards_, dones_, actions_, old_action_probs_, old_state_estimates_, epinfos = [], [], [], [], [], [], []
        
        
        for s in range(n_steps):
            frames_.append(frame)
            action_probs, state_estimate = agent(np_to_pytorch_img(frame).to(device))
            action = get_action(action_probs).cpu().numpy()
            frame, reward, done, info = env.step(action)
            rewards_.append(reward)
            dones_.append(done)
            actions_.append(action)
            old_action_probs_.append(action_probs.detach().cpu().numpy())
            old_state_estimates_.append(state_estimate.detach().cpu().numpy())

            for i_info in info:
                episode_info = i_info.get('episode')
                if episode_info: epinfos.append(episode_info)

        rewards_ = np.array(rewards_)
        dones_ = np.array(dones_)
        actions_ = np.array(actions_)
        frames_ = np.array(frames_)
        old_state_estimates_ = np.array(old_state_estimates_)
        
        last_frame_ = frames_[-1] # reset global last frame. Next time we gather trajectories we'll pick up here
        
        rewards_[-1] = state_estimate.squeeze(-1).cpu().numpy() # Bootstrapped returns

        returns_ = compute_returns_gae(rewards_, dones_, old_state_estimates_.squeeze(-1))
        
        # Reshaping dims and prepping tensor types and locations. Nothing conceptually interesting.
        returns_, old_state_estimates_, old_action_probs_, actions_, frames_ = reshaping_processing_acrobatics(returns_, 
                                                                                                   old_state_estimates_, 
                                                                                                   old_action_probs_, 
                                                                                                   actions_, frames_)
        
        returns, frames, actions, old_action_probs, old_state_estimates = (returns_, 
                                                                           frames_, 
                                                                           actions_, 
                                                                           old_action_probs_, 
                                                                           old_state_estimates_)
        
        avg_score = get_avg_score(epinfos)
        scores.append(avg_score)
        total_steps += n_steps
    '''train agent for n_opt_epochs'''
    
    for e in range(n_opt_epochs):
        
        """shuffle_database"""
        
        dataset_ix = torch.randperm(len(returns));
        returns = returns[dataset_ix]
        frames = frames[dataset_ix]
        actions = actions[dataset_ix]
        old_action_probs = old_action_probs[dataset_ix]
        old_state_estimates = old_state_estimates[dataset_ix]
        
        """ Learn from current database of rollouts for a single epoch """
        
        agent.train()
        epoch_losses = []
        ix_range = range(len(returns))
        ix = 0
        grad_accum_counter = 1
        while ix < len(returns):
            batch_ix = ix_range[ix:ix+bs]; ix += bs
            entropy_bonus, action_gain, critic_loss = run_batch(agent, 
                                                                batch_ix,
                                                                device,
                                                                frames,
                                                                returns, 
                                                                actions, 
                                                                old_action_probs,
                                                                old_state_estimates, 
                                                                verbose=verbose)
            entropy_bonus *= entropy_coef
            critic_loss *= value_loss_coef
            if verbose: print("\n\nentropy bonus, action gain\n, critic loss\n",
                              entropy_bonus.item(),
                              action_gain.item(),
                              critic_loss.item())
            
            total_loss = critic_loss - entropy_bonus - action_gain
            total_loss /= accumulation_steps
            total_loss.backward()

            if grad_accum_counter % accumulation_steps == 0:
                nn.utils.clip_grad_norm_(agent.parameters(), .5);
                opt.step()
                opt.zero_grad()
            grad_accum_counter+=1
            epoch_losses.append(total_loss.item())
        loss = np.array(epoch_losses).mean()
        
        """append loss"""
        losses.append(loss)
    
    """evaluate agent"""
    
    for eval_env_index in range(args.start_level_min, env_index):
        eval_score = evaluate_agent(agent, envs[eval_env_index], eval_steps, device)
        writer.add_scalar('avg_score/eval_env' + str(eval_env_index), eval_score, total_steps)
    
    """save model and write logs"""
    avg_loss = sum(losses[-n_opt_epochs:])/len(losses[-n_opt_epochs:])
    print("round:{:>4},  avg score:{:>6.2f},  avg loss:{:>6.2f},  env index:{:>2},  total steps:{:>6}".format(i,
                                                                                  round(float(avg_score), 2),
                                                                                  round(avg_loss, 2),
                                                                                  env_index,
                                                                                  total_steps))
    writer.add_scalar('avg_score/total', avg_score, total_steps)
    writer.add_scalar('avg_loss', avg_loss, total_steps)
    
    torch.save(agent.state_dict(), save_path)