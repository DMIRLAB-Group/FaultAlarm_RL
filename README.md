# FaultAlarm_RL
Official code for paper [Learning by Doing: An Online Causal Reinforcement Learning Framework with Causal-Aware Policy](https://arxiv.org/abs/2402.04869)

## Installations


### Create Conda Environment

```
conda create -n faultAlarm_RL python=3.8
source activate faultAlarm_RL
pip install torch==1.7.1+cu110 -f https://download.pytorch.org/whl/torch_stable.html
```

### Install requirements

```commandline
git clone https://github.com/DMIRLAB-Group/FaultAlarm_RL.git
cd ./FaultAlarm_RL
pip install -r requirements.txt
```

### Install FaultAlarm RL Environment

```commandline
cd ./FaultAlarm_RL
pip install -e .
```
This will install the custom faultAlarm environment so it can be accessed via standard Python imports.

## Usage
Below is a simple example demonstrating how to interact with the faultAlarm environment via gym:
### Usage for FaultAlarm RL Environment

```
import gym
import faultAlarm


env = gym.make('FaultAlarm-v0')
print("state space dimension : ", env.observation_space.shape[0])
print("action space dimension : ", env.action_space.n)

state = env.reset()
while True:
    action = env.action_space.sample()
    
    next_state, reward, done, info = env.step(action)
    
    state = next_state
    if done:
         break
```


## Run 

### Logging
This codebase includes TensorBoard logging. To visualize training logs, run:
```
tensorboard --logdir=logs_tensorboard --port 6008
```
assuming you used the default log_dir.

### RL method
An example of training a standard PPO agent on the FaultAlarm environment:


```python
python train_ppo.py --max_episodes 2000 --max_ep_len 100
```


### CausalRL method

An example of training a Causal PPO agent on the FaultAlarm environment:

```python
python train_c_ppo.py --subset_size 8 --random_g 0 --reg_parm 0.005
```
Where:

- subset_size: Specifies the size of the causal action subsets.
- random_g: Determines whether to initialize the agent with a random causal graph.
- reg_parm: A hyperparameter for causal pruning when learning causal structures.


Below is an overview of the main components related to causal reinforcement learning:
```
└── CausalReinforcementLearning
    ├── causal_learner.py
    |   │   ├──THP: Used to learn initial causal structure
    │   │   ├──CausalLearner: Used for online causal structure learning, including orientation and pruning stages.
    ├── c_ppo.py
```

