import gym
import faultAlarm
from gym import wrappers
from faultAlarm.envs.faultAlarm import FaultAlarmEnv
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import numpy as np
import random
import torch
from ReinforcementLearning.ppo import PPO

import argparse
import os

parser = argparse.ArgumentParser(description="data")
# training
parser.add_argument('-c', '--cuda', type=int, default=0)
parser.add_argument('-s', '--seed', type=int, default=40)
parser.add_argument('-me', '--max_episodes', type=int, default=2000)
parser.add_argument('-mel', '--max_ep_len', type=int, default=100)

args = parser.parse_args()
max_episodes = args.max_episodes
max_ep_len = args.max_ep_len
cuda = args.cuda
random_seed = args.seed

hyperparameters = {
    "PPO": {
        "gamma": 0.99,  # discount factor
        "hyperparameterstau": 0.005,  # soft update target network
        "lr_actor": 0.0003,  # learning rate for actor network
        "lr_critic": 0.0003,  # learning rate for critic network
        "eps_clip": 0.2,  # clip parameter for PPO
        "begin_update_timestep": 512,
        "update_timestep": 512,  # update policy every n timesteps
        "K_epochs": 50,  # update policy for K epochs in one PPO update
        "batch_size": 64,
        "hidden_units": 128,
        "action_std_init": 0.6,
        "has_continuous_action_space": False,
        "max_grad_norm": 0.5, 
        "normalize_advantage": True 
    }
}

algorithm = 'PPO'
agent_class = PPO
agent_config = hyperparameters[algorithm]

# load env
env = FaultAlarmEnv()
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n  # discrete action

# set device to cpu or cuda
device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda:' + str(cuda))
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")

print("setting random seed to ", random_seed)
torch.manual_seed(random_seed)
env.seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

print("algo : ", algorithm)
print("state space dimension : ", state_dim)
print("action space dimension : ", action_dim)
print("max_episodes : ", max_episodes)
print("max_ep_len : ", max_ep_len)
print("cuda : ", cuda)


# tensorboard
# save logs
output_dir = "logs_tensorboard"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
dir_name = f"./{output_dir}/env_real_{algorithm}_seed_{random_seed}_ep_{max_episodes}"
writer = SummaryWriter(dir_name)

# init agent
agent = agent_class(state_dim=state_dim,
                    action_dim=action_dim,
                    config=agent_config,
                    device=device)

i_episode, i_update, time_step = 1, 0, 0

# training loop
while i_episode <= max_episodes:
    
    state = env.reset()

    ep_rewards = 0  # Record the cumulative rewards for each episode
    ep_steps = 0  # Record each episode steps
    ep_cum_alarms = 0  # Record the cumulative alarms for each episode

    for t in range(1, max_ep_len + 1):
        # select action
        action = agent.select_action(state)

        # interact with the environment
        state_, reward, done, info = env.step(action)
        ep_cum_alarms += info["alarm_num"]
        ep_rewards += reward
        
        # saving experience
        agent.buffer.add(state, action, reward, state_, done)

        state = state_

        time_step += 1
        ep_steps += 1

        # update agent
        if time_step > agent_config["begin_update_timestep"]:
            # Update policy by n steps
            if time_step % agent_config["update_timestep"] == 0:
                loss, pg_loss, value_loss, entropy_loss = agent.update()
                i_update += 1

        # print info
        if time_step % 2000 == 0:
            print("---already update ", i_update, " times! and interact ", time_step, " times!---")

        if done:  # break; if the episode is over
            break

    ep_avg_alarms = int(ep_cum_alarms / ep_steps) if ep_cum_alarms != 0 else ep_avg_alarms
   
    writer.add_scalar('Reward/Reward', ep_rewards, i_episode)
    writer.add_scalar('Steps/Steps', ep_steps, i_episode)
    writer.add_scalar('Alarms/Avg Alarms', ep_avg_alarms, i_episode)
    
    
    print("Episode: ", i_episode, 
        " reward: ", round(ep_rewards, 2),
        " ep_steps: ", ep_steps,
        " avg alarm nums: ", ep_avg_alarms)

    i_episode += 1

writer.close()



