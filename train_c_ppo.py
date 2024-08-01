import gym
import faultAlarm
from gym import wrappers
from faultAlarm.envs.faultAlarm import FaultAlarmEnv

import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import numpy as np
import random
import torch
from CausalReinforcementLearning.c_ppo import C_PPO
from CausalReinforcementLearning.causal_learner import CausalLearner
import argparse
import networkx as nx
import pickle
import os
from graphviz import Digraph


def plot_g(nodes, g_mat, name):
    dag_2 = Digraph("event", format='png')
    dag_2.attr('node', shape='circle')
    node_name_arr = []
    # 画点
    for i in nodes:
        node_name = str(i)
        dag_2.node(node_name, node_name)
    # 画dag
    for i in nodes:
        for j in nodes:
            if g_mat[i, j] != 0:
                dag_2.edge(str(i), str(j))

    dag_2.render(filename=name, view=False)
    

parser = argparse.ArgumentParser(description="data")
# training
parser.add_argument('-c', '--cuda', type=int, default=0)
parser.add_argument('-s', '--seed', type=int, default=40)
parser.add_argument('-me', '--max_episodes', type=int, default=1000)
parser.add_argument('-mel', '--max_ep_len', type=int, default=100)

parser.add_argument('-sz', '--subset_size', type=int, default=8)
parser.add_argument('-reg', '--reg_parm', type=float, default=0.005)

parser.add_argument('-rd', '--random_g', type=int, default=0)

args = parser.parse_args()

max_episodes = args.max_episodes
max_ep_len = args.max_ep_len
cuda = args.cuda
random_seed = args.seed

subset_size = args.subset_size
reg_parm = args.reg_parm

random_g = args.random_g

begin_steps = 1024

hyperparameters = {
    "C_PPO": {
        "gamma": 0.99,  # discount factor
        "lr_actor": 0.0003,  # learning rate for actor network
        "lr_critic": 0.0003,  # learning rate for critic network
        "eps_clip": 0.2,  # clip parameter for PPO
        "eps_causal": 0.2,
        "update_timestep": 256,  # update policy every n timesteps
        "K_epochs": 50,  # update policy for K epochs in one PPO update
        "batch_size": 64,
        "hidden_units": 128,
        "hidden_size": 128,
        "begin_update_timestep": 512,
        "is_update_by_single_step": False,
        "has_continuous_action_space": False,
        "clip_grad_param": 0.5,  # 对梯度的clip
        "is_clip_grad": True,
        "reward_scale": 1.0,
        "update_mode": "soft",
        "tau": 0.005,  # soft update target network
        "target_hard_update": 100,
        "normalize_advantage": True  # 对标准化
    }
}
algorithm = 'C_PPO'
agent_class = C_PPO
agent_config = hyperparameters[algorithm]

model_path = f'faultAlarm/EnvModel/real_model.pkl'
env = FaultAlarmEnv(model_path=model_path,return_obs=True)

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
if random_seed:
    print("setting random seed to ", random_seed)
    torch.manual_seed(random_seed)
    env.seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

print("algorithm : ", algorithm)
print("cuda : ", cuda)
print("state space dimension : ", state_dim)
print("action space dimension : ", action_dim)
print("max_episodes : ", max_episodes)
print("max_ep_len : ", max_ep_len)
            
# tensorboard
output_dir = "logs_tensorboard"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
dir_name = f"./{output_dir}/env_real_{algorithm}_seed_{random_seed}_ep_{max_episodes}_rg_{str(random_g)}_reg_{str(reg_parm)}"
print("writing in ", dir_name)
writer = SummaryWriter(dir_name)

if random_g == 1:
    agent_config["eps_causal"] = 0.6
else:
    agent_config["eps_causal"] = 0.3

causal_begin_steps = agent_config["begin_update_timestep"]
            
# init agent
agent = agent_class(state_dim=state_dim,
                    action_dim=action_dim,
                    config=agent_config,
                    device=device)

dataset_path = f'faultAlarm/data/synthetic_data/real_model_sample_size_4000_seed_{random_seed}'
causal_learner = CausalLearner(env.node_num, env.event_num, subset_size, dataset_path, random_g,random_seed)

# causal learning
node_num, event_num = causal_learner.node_num, causal_learner.event_num
causal_info = causal_learner.learn_causal_graph()
max_hop = causal_info["max_hop"]
event_order, subset_size = causal_info["event_order"], causal_info["subset_size"]
alpha_mat, mu, edge_mat = causal_info["alpha_mat"], causal_info["mu"], causal_info["edge_mat"]
init_edge_mat = edge_mat.copy()
topology = env.topology
topology_mat = nx.to_scipy_sparse_array(topology).todense().astype('float')
topo_k = np.zeros([node_num, node_num])
for k in range(max_hop):
    topo_k += topology_mat ** k
topology_mat = np.array(topology_mat)
# node order based on topology graph
neighbor = list()
node_order = []
for i in range(node_num):
    neighbor.append((sum(topology_mat[i]), i))
neighbor.sort(reverse=True)
for j in range(len(neighbor)):
    node_order.append(int(neighbor[j][1]))

intervention_mat = np.zeros((event_num, event_num, 3))
avg_intervention_mat = np.zeros((event_num, event_num))
# training loop
i_episode, i_update, time_step = 1, 0, 0
i_causal_update = 0
precision,recall,f1 = causal_info['precision'],causal_info['recall'],causal_info['f1']
print(f"THP result, F1 {f1}, Precision {precision}, Recall {recall}")
new_f1, new_precision, new_recall = f1, precision,recall

thp_new_alpha_mat = alpha_mat.copy()
thp_new_edge_mat = edge_mat.copy()
writer.add_scalar('CausalResult/precision', precision, i_causal_update)
writer.add_scalar('CausalResult/recall', recall, i_causal_update)
writer.add_scalar('CausalResult/f1', f1, i_causal_update)

writer.add_scalar('CausalResult/epi_precision', precision, i_episode)
writer.add_scalar('CausalResult/epi_recall', recall, i_episode)
writer.add_scalar('CausalResult/epi_f1', f1, i_episode)

while i_episode <= max_episodes:
    obs, state = env.reset()

    ep_rewards = 0  # Record the cumulative rewards for each episode
    ep_cf_reward = 0
    ep_new_reward = 0
    ep_steps = 0  # Record each episode steps
    ep_cum_alarms = 0  # Record the cumulative alarms for each episode
    ep_avg_alarms = 0

    for t in range(1, max_ep_len + 1):
        current_time = env.current_time
        
        action, action_prob = agent.select_action(state, node_order, event_order, subset_size)

        # interact with the environment
        obs_, state_, reward, done, info = env.step(action)
        ep_cum_alarms += info["alarm_num"]
        ep_rewards += reward

        # saving experience
        agent.buffer.add(state, action, reward, state_, done)

        # Record the change in the number of alarms for child events before and after the intervention
        node = int(action / event_num)
        event_type = int(action % event_num)

        intervention_mat[event_type, :, 2] += 1

        # find neighbours base on topology mat
        neighbours = set(np.array(np.nonzero(topo_k[node])).flatten())
        neighbours.add(node)

        possible_child_events = set()
        for j in range(event_num):
            for i in neighbours:
                if (i, j) != (node, event_type):
                    possible_child_events.add((i, j))
        before_do = np.zeros(event_num)
        after_do = np.zeros(event_num)
        for row in obs.iterrows():
            n = row[1]['Node']
            e = row[1]['Event']
            if (n, e) in possible_child_events:
                before_do[e] += 1

        for row in obs_.iterrows():
            n = row[1]['Node']
            e = row[1]['Event']
            e_start_time = row[1]['Start Time Stamp']
            if (n, e) in possible_child_events:
                after_do[e] += 1

        # event_type -> k, k is every possible event type
        for k in range(event_num):
            if k == event_type:
                intervention_mat[event_type, k, 1] += 1
                continue
            e_num = before_do[k] - after_do[k]
            intervention_mat[event_type, k, 0] += e_num
            intervention_mat[event_type, k, 1] += 1

        # ---------------------------------- update edges ------------------------------------------------------------
        if intervention_mat[event_type, 0, 1] >= 3:
    
            is_update = False
            i = event_type

            avg_intervention_mat[i,:] = intervention_mat[i, :, 0] / intervention_mat[i, :, 1]
            avg_intervention_vector = avg_intervention_mat[i,:].copy()
            max_index = np.argmax(avg_intervention_vector)

            # remove edge first
            remove_index = np.where(avg_intervention_vector<0)[0]
            for r in remove_index:
                if edge_mat[i, r] != 0:
                    edge_mat[i, r] = 0
                    is_update = True

            # then add edge
            add_index = np.flip(np.argsort(avg_intervention_vector))
            count_edge = len(np.nonzero(edge_mat[i, :])[0])
            for j in add_index:
                # remove edge
                if avg_intervention_mat[i, j] < 0 and init_edge_mat[i, j] == 0 and edge_mat[i, j] != 0:
                    edge_mat[i, j] = 0
                    is_update = True
                
                # add edge
                if avg_intervention_mat[i, j] > 0 and edge_mat[i, j] == 0:
                    edge_mat[i, j] = 1
                    is_update = True
                    edge_matrix = np.matrix(edge_mat)
                    # check the new graph is a DAG or not
                    g = nx.from_numpy_array(edge_matrix)
                    is_acyclic = nx.is_directed_acyclic_graph(g)
                    
                    if is_acyclic:
                        edge_mat[i, j] = 0
                        is_update = False

            if is_update:
                
                i_causal_update += 1
                recall, precision, f1 = causal_learner.get_performance(edge_mat, env.edge_mat)
                
                thp_new_edge_mat, thp_new_alpha_mat, new_f1, new_precision, new_recall = causal_learner.update_edge_mat(edge_mat.copy(),reg_parm)
                
                f1,precision, recall = new_f1, new_precision, new_recall
                
                thp_causal_order = causal_learner.estimate_causal_order(thp_new_edge_mat.copy())
                thp_causal_order = np.flipud(thp_causal_order)
                thp_event_order = thp_causal_order

                causal_order = causal_learner.estimate_causal_order(edge_mat.copy())
                causal_order = np.flipud(causal_order)
                event_order = causal_order

                if i_causal_update >15:
                    event_order = thp_event_order
                if i_causal_update % 10 == 0:
                    print(f'i_causal_update {i_causal_update}, f1 {f1},precison {precision},recall {recall}')
                
                writer.add_scalar('CausalResult/precision', precision, i_causal_update)
                writer.add_scalar('CausalResult/recall', recall, i_causal_update)
                writer.add_scalar('CausalResult/f1', f1, i_causal_update)

        state = state_
        obs = obs_

        time_step += 1
        ep_steps += 1

        # update agent
        if time_step > agent_config["begin_update_timestep"]:
            if time_step % agent_config["update_timestep"] == 0:
                loss, pg_loss, value_loss, entropy_loss = agent.update()
                i_update += 1
       
        if done:  # break; if the episode is over
            break
    
    
    ep_avg_alarms = int(ep_cum_alarms / ep_steps) if ep_cum_alarms != 0 else ep_avg_alarms

    causal_order = causal_learner.estimate_causal_order(edge_mat.copy())
    causal_order = np.flipud(causal_order)
    if (i_episode-1) % 20==0:
        print("Episode: ", i_episode,
                " reward: ", round(ep_rewards, 2),
                " ep_steps: ", ep_steps,
                " avg alarm nums: ", ep_avg_alarms,
                " f1: ", round(new_f1, 4),
                )
    

    writer.add_scalar('Reward/Reward', ep_rewards, i_episode)
    writer.add_scalar('Steps/Steps', ep_steps, i_episode)
    writer.add_scalar('Alarms/Avg Alarms', ep_avg_alarms, i_episode)

    writer.add_scalar('CausalResult/epi_precision', new_precision, i_episode)
    writer.add_scalar('CausalResult/epi_recall', new_recall, i_episode)
    writer.add_scalar('CausalResult/epi_f1', new_f1, i_episode)
    
    i_episode += 1

writer.close()

