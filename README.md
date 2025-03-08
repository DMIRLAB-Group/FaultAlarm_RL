# FaultAlarm_RL
Official code for paper [Learning by Doing: An Online Causal Reinforcement Learning Framework with Causal-Aware Policy](https://arxiv.org/abs/2402.04869)

## Installations

Make sure you have install Pytorch.

Install requirements

```commandline
pip install -r requirements.txt
```

Install Environment

```commandline
pip install -e .
```


## Usage for Reinforcement Learning

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
## Examples 
An example run of the CausalPPO algorithm on FaultAlarm environment is as follows.

```python

python train_c_ppo.py
```

