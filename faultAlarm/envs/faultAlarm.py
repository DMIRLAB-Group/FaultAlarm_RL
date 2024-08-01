import gym
from gym import error, spaces, utils
from gym.utils import seeding
import pandas as pd
import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt
from itertools import product
import pickle
import os
from graphviz import Digraph

pd.options.mode.chained_assignment = None  # default='warn'
"""
This is a fault alarm simulation environment.
"""

class FaultAlarmEnv(object):
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 model_path='faultAlarm/EnvModel/real_model.pkl',
                 return_obs=False
                ):
        
        self.env_name = 'faultAlarm-v0'
        
        # load environment model
        model = self.load_env_model(model_path)
       
        self.node_num = model["node_num"]
        self.event_num = model["event_num"]
        self.rootcause_num = model['rootcause_num']
        self.init_rc_num = model["init_rc_num"]
        self.time_range = model["time_range"]
        self.alarm_type_arr = model["alarm_type_arr"]
        self.alarm_type_ratio = model["alarm_type_ratio"]
        self.init_alarm_num = model["init_alarm_num"]
        self.max_alarms = model["max_alarms"]
        self.graph = model["graph"]
        self.edge_mat = model['edge_mat']
        self.topology = model["topology"]
        self.max_hop = model["max_hop"]
        self.alpha = model["alpha"]
        self.alpha_mat = model["alpha"]
        self.mu = model["mu"]
        self.A_sys_k = model["A_sys_k"]
        self.causal_order = model["causal_order"]
        self.topo_order = model["topo_order"]     
        self.root_nodes_num = model["root_nodes_num"]
        self.root_events_num = model["root_events_num"] 
        self.poss_root_alarms = model["poss_root_alarms"] 
        self.limit_same_alarms= model['limit_same_alarms']
        self.topology_nodes = np.arange(self.node_num)    

        # whether return alarms table
        self.return_obs = return_obs

        # Define action and observation space
        self.action_space = spaces.Discrete(self.node_num * self.event_num)
        self.observation_space = spaces.Box(low=0, high=100000, shape=(self.node_num * self.event_num * 2,), dtype=np.uint8)
        self.reward_range = [-2, 2]

        self.current_time = 0
        self.current_state = np.zeros((self.node_num, self.event_num, 2)).flatten()
        self.current_observation = pd.DataFrame(
            columns=['Alarm Id', 'Node', 'Event', 'Start Time Stamp',
                     'Last Time Stamp', 'Flag', 'Father Id', 'Father Node', 'Father Event', 'Type'])

        self.alarm_id = 0  # record an alarm
        self.alarm_table = pd.DataFrame(
            columns=['Alarm Id', 'Node', 'Event', 'Start Time Stamp', 'Last Time Stamp', 'Flag',
                     'Father Id', 'Father Node', 'Father Event', 'Type'])

        self.steps = 0
        self.max_steps = 100
        self.min_steps = self.init_rc_num * 2
    
    def load_env_model(self, model_path):
        # load trained env model
        model = pickle.load(open(model_path, 'rb'))
        return model

    def reset(self):
        # reset the system
        self.steps = 0
        self.current_time = 0
        self.alarm_id = 0
        self.alarm_table.drop(index=self.alarm_table.index, inplace=True)
        self.current_observation.drop(index=self.current_observation.index, inplace=True)
        self.current_state = np.zeros((self.node_num, self.event_num, 2)).flatten()
        
        # generate root alarms 
        root_alarms_table = self._generate_root_alarms(root_alarms_num=self.init_rc_num, is_init=True)

        self.alarm_table = pd.concat([self.alarm_table, root_alarms_table])

        self.current_observation = pd.concat([self.current_observation, root_alarms_table])
        self.current_state = self._get_state(self.current_observation)

        self.current_time = root_alarms_table['Start Time Stamp'].max() + 1

        observation = self.current_observation
        state = self._get_state(observation)

        while len(observation) < self.init_alarm_num:
            observation, new_num, noise_num, n_root_alarms = self._generate_alarms(
                                                    self.current_observation,
                                                    is_init=True)
            self.current_time += self.time_range
            self.current_observation = observation
            state = self._get_state(observation)
            self.current_state = state

        if self.return_obs:
            return observation, state
        else:
            return state

    def step(self, action):
        """
        :param action: list
        :return: next_state, reward
        """
        self.steps += 1
        done = False
        if action is None:

            next_observation, new_num, noise_num, n_root_alarms = self._generate_alarms(
                self.current_observation)

            reward = self._reward_func(next_observation)
            next_state = self._get_state(next_observation)

            self.current_observation = next_observation
            self.current_state = next_state
            self.current_time += self.time_range
            if len(next_observation) < 0 or len(next_observation) > self.max_alarms:
                done = True
            else:
                done = False
            info = {"new_alarm_num": new_num,
                    "noise_num": noise_num,
                    "alarm_num": len(next_observation),
                    "root_num": n_root_alarms}
            if self.return_obs:
                return next_observation, next_state, reward, done, info
            else:
                return next_state, reward, done, info

        # do alarm(node,event_type), remove alarm and its sub-alarms
        node = int(action / self.event_num)
        event_type = action % self.event_num

        self.alarm_table.reset_index(drop=True, inplace=True)
        self.current_observation.reset_index(drop=True, inplace=True)

        do_num = len(self.current_observation[
                (self.current_observation['Node'] == node) & (self.current_observation['Event'] == event_type)])

        # find alarm index and sub-alarms index
        remove_index = self._find_remove_index(node, event_type, self.current_observation)
        
        # remove alarms
        observation = self.current_observation.drop(index=remove_index).reset_index(drop=True)
        
        if len(observation) == 0:
            done = True
            next_observation = observation
            new_num, noise_num, n_root_alarms = 0,0,0
        else:
            # generate next observation
            next_observation, new_num, noise_num, n_root_alarms = self._generate_alarms(
                observation)

        next_state = self._get_state(next_observation)
        reward = self._reward_func(next_observation, do_num)
        info = {"new_alarm_num": new_num,
                "noise_num": noise_num,
                "alarm_num": len(next_observation),
                "root_num": n_root_alarms}

        self.current_observation = next_observation
        self.current_state = next_state
        self.current_time += self.time_range

        if len(next_observation) == 0 or len(next_observation) > self.max_alarms:
            done = True
        
        if self.return_obs:
            return next_observation, next_state, reward, done, info
        else:
            return next_state, reward, done, info

    def _reward_func(self, next_observation, do_num=1):
        """
        reward =  ((m_t - m_next_t) / m_t) - steps
        range = [-1,1]+[-1,-0.01] = [-2,0.99]
        """
        m_t = len(self.current_observation) - do_num
        m_next_t = len(next_observation)
        c_1 = 1
        c_2 = 1 / self.max_steps
        r_1 = 1
        r_2 = self.steps 
        if m_next_t == 0 or m_t == 0:
            r_1 = 1
        else:
            r_1 = (m_t - m_next_t) / m_t
        
        reward = r_1 * c_1 - r_2 * c_2
        
        return round(reward, 2)

    def _generate_root_alarms(self, root_alarms_num, is_init=False):
        """
        Generate root cause events based on cause graph and topology graph
        :return: root_alarms_table
        """
        root_alarms_table = pd.DataFrame(
                columns=['Alarm Id', 'Node', 'Event', 'Start Time Stamp', 'Last Time Stamp', 'Flag',
                                'Father Id', 'Father Node', 'Father Event', 'Type'])
        if root_alarms_num == 0:
            return root_alarms_table
        
        truncate = 500 if is_init else self.current_time + self.time_range
        start_time = 0 if is_init else self.current_time

        node_order = self.topo_order.copy()
        root_nodes = node_order[:self.root_nodes_num]

        # root cause
        poss_root_alarms = self.poss_root_alarms.copy()

        count_alarm_num = 0
        while count_alarm_num < root_alarms_num:

            if len(poss_root_alarms) == 0:
                poss_root_alarms = self.poss_root_alarms.copy()
            (node, event) = random.sample(poss_root_alarms, 1)[0]
            
            alarm_start_time = start_time + np.random.exponential(1 / (1 * self.mu[event]))

            if alarm_start_time < truncate:
                if len(poss_root_alarms) == 0:
                    poss_root_alarms = self.poss_root_alarms.copy()
                else:
                    for n in root_nodes:
                        poss_root_alarms.remove((n,event))

                count_alarm_num += 1
                # flag==0 indicates root cause alarm
                flag, father_id, father_node, father_event = 0, -1, -1, -1

                alarm_type = 0
                alarm_last_time = 999999

                self.alarm_id += 1

                new_e = np.array([(self.alarm_id, node, event, alarm_start_time, alarm_last_time, flag,
                                father_id, father_node, father_event, alarm_type)]).astype('int')

                new_e = pd.DataFrame(new_e, columns=['Alarm Id', 'Node', 'Event', 'Start Time Stamp',
                                                    'Last Time Stamp', 'Flag', 'Father Id', 'Father Node',
                                                    'Father Event', 'Type'])
                root_alarms_table = pd.concat([root_alarms_table, new_e])

        root_alarms_table = root_alarms_table.sort_values(
            ['Start Time Stamp', 'Node']).reset_index(drop=True)

        return root_alarms_table

    def _generate_noise(self, noise_alarm_num, t, truncate, is_init=False):

        noise_alarms_table = pd.DataFrame(
            columns=['Alarm Id', 'Node', 'Event', 'Start Time Stamp', 'Last Time Stamp', 'Flag',
                        'Father Id', 'Father Node', 'Father Event', 'Type'])
        if noise_alarm_num == 0:
            return noise_alarms_table
            
        event_order = self.causal_order.copy()
        node_order = self.topo_order.copy()
        root_events = event_order[:self.root_events_num]
        node_order = self.topo_order.copy()
        noise_nodes = node_order[self.root_nodes_num:]
        noise_mu = self.mu.copy()
        for i in root_events:
            noise_mu[i] = 0
        noise_node = np.random.choice(noise_nodes, 2)
        for v in noise_node:
            start_t = 0 if is_init else t
            noise_alarms = self._get_alarm_list(node=v,
                                                intensity=noise_mu,
                                                start_time=start_t,
                                                truncate=truncate,
                                                flag=2,
                                                father_id=-1,
                                                father_node=-1,
                                                father_event=-1,
                                                strength=1)
            if len(noise_alarms) == 0:
                continue

            new_e = np.array(noise_alarms).astype('int')
            new_e = pd.DataFrame(new_e, columns=['Alarm Id', 'Event', 'Start Time Stamp', 'Last Time Stamp', 'Flag',
                                                    'Father Id', 'Father Node', 'Father Event', 'Type'])
            new_e['Node'] = v
            noise_alarms_table = pd.concat([noise_alarms_table, new_e])
            
        noise_alarms_table.reset_index(drop=True, inplace=True)
        noise_alarm_num = len(noise_alarms_table) if len(noise_alarms_table)< noise_alarm_num else noise_alarm_num
        noise_alarms_table = noise_alarms_table.sample(n=noise_alarm_num, replace=True, random_state=1)

        return noise_alarms_table

    def _generate_alarms(self, historical_alarms, is_init=False):

        t = self.current_time
        time_range = self.time_range
        truncate = 350 if is_init else t + time_range
        max_alarm_num = self.init_alarm_num if is_init else self.max_alarms
        
        # randomly generate root alarm
        n_root_alarms = np.random.choice([0,1], p=[0.8, 0.2], size=(1))[0]
        root_alarms_table = self._generate_root_alarms(root_alarms_num=n_root_alarms, is_init=False)

        # randomly generate noise alarm
        noise_alarms_num = np.random.choice([0,1], p=[0.9, 0.1], size=(1))[0]
        noise_alarms_table = self._generate_noise(noise_alarms_num, t, truncate, is_init)
        
        # base_alarms ==> historical_alarms + noise_alarms
        base_alarms_table = pd.concat([historical_alarms, root_alarms_table, noise_alarms_table])

        # Sub-alarms are generated by base alarms propagation
        sub_alarms_table, is_max_alarm = self._generate_sub_alarms(t, base_alarms_table, truncate, max_alarm_num, is_init)

        new_alarms_table = pd.concat([root_alarms_table, noise_alarms_table, sub_alarms_table])

        historical_alarms['End Time Stamp'] = historical_alarms['Start Time Stamp'] + historical_alarms[
            'Last Time Stamp']
        end_alarms = historical_alarms[historical_alarms['End Time Stamp'] < t]
        remove_index = []
        for row in end_alarms.iterrows():
            node = row[1]['Node']
            event = row[1]['Event']
            alarm_id = row[1]['Alarm Id']
            remove_index += self._find_remove_index(node, event, historical_alarms, alarm_id, deleteById=True)

        historical_alarms = historical_alarms.drop(index=remove_index).reset_index(drop=True)

        still_alarms = historical_alarms[historical_alarms['End Time Stamp'] >= t]
        still_alarms = still_alarms.iloc[:, :-1]
        obs_alarm_table = pd.concat([still_alarms, new_alarms_table])
        obs_alarm_table = obs_alarm_table.sort_values(
            ['Start Time Stamp', 'Node']).reset_index(drop=True)

        new_num = len(new_alarms_table)

        # save to alarm table
        self.alarm_table = pd.concat([self.alarm_table, new_alarms_table])
        self.alarm_table = self.alarm_table.sort_values(
            ['Start Time Stamp', 'Node']).reset_index(drop=True)

        return obs_alarm_table, new_num, noise_alarms_num, n_root_alarms

    def _generate_sub_alarms(self, t, base_alarms_table, truncate, max_alarm_num, is_init):
        """
        base_alarms_table -> sub_alarms_table
        """
        is_max_alarm = False

        sub_alarms_table = pd.DataFrame(
            columns=['Alarm Id', 'Node', 'Event', 'Start Time Stamp', 'Last Time Stamp', 'Flag',
                     'Father Id', 'Father Node', 'Father Event', 'Type'])

        base_alarms = dict()
        n = list(self.topology_nodes)
        for v in n:
            base_alarms[v] = []
        for row in base_alarms_table.iterrows():
            alarm_id = row[1]['Alarm Id']
            node = row[1]['Node']
            event = row[1]['Event']
            start_time = row[1]['Start Time Stamp']
            last_time = row[1]['Last Time Stamp']
            flag = row[1]['Flag']
            father_id = row[1]['Father Id']
            father_node = row[1]['Father Node']
            father_event = row[1]['Father Event']
            alarm_type = row[1]['Type']

            base_alarms[node].append(
                (alarm_id, event, start_time, last_time, flag, father_id, father_node, father_event, alarm_type))

        current_alarm_num = len(base_alarms_table)
        while sum(map(len, base_alarms.values())) != 0:

            new_alarms_dict = dict()
            np.random.shuffle(n)

            for v in n:
                new_alarms_dict[v] = []
                for k in range(self.max_hop):
                    for neighbour in self.topology_nodes:
                        strength = self.A_sys_k[k, v, neighbour]  
                        if strength == 0:
                            continue
                    
                        for i in base_alarms[neighbour]:
                            if current_alarm_num > max_alarm_num:
                                break
                            # i: (alarm_id, event, start_time, last_time, flag, father_id, father_node, father_event, alarm_type)
                            trun = truncate if i[8] == 0 else (i[2] + i[3])
                            # flag==0 means it is a root cause event, flag==1 means it is a propagated sub-event
                            flag = 3 if i[4] == 2 else 1
                            start_time = i[2] if is_init else t
                            new_alarms = self._get_alarm_list(node=v,
                                                              intensity=self.alpha[k, i[1]],
                                                              start_time=start_time,
                                                              truncate=trun,
                                                              flag=flag,
                                                              father_id=i[0],
                                                              father_node=neighbour,
                                                              father_event=i[1],
                                                              strength=strength)
                            new_alarms_dict[v] += new_alarms
                            current_alarm_num += len(new_alarms)

                if len(new_alarms_dict[v]) == 0:
                    continue

                new_e = np.array(new_alarms_dict[v]).astype('int')
                new_e = pd.DataFrame(new_e, columns=['Alarm Id', 'Event', 'Start Time Stamp',
                                                     'Last Time Stamp', 'Flag', 'Father Id', 'Father Node',
                                                     'Father Event', 'Type'])
                new_e['Node'] = v

                sub_alarms_table = pd.concat([sub_alarms_table, new_e])

                if current_alarm_num > max_alarm_num:
                    is_max_alarm = True
                    return sub_alarms_table, is_max_alarm

            base_alarms = new_alarms_dict

        if current_alarm_num > max_alarm_num:
            is_max_alarm = True

        return sub_alarms_table, is_max_alarm

    def _get_alarm_list(self, node, intensity, start_time, truncate,
                        flag=0, father_id=-1, father_node=-1, father_event=-1,
                        strength=1):
        """
        kernel function: kappa=a*exp(-a*t)
        """
        sub_alarm_list = []
       
        e = np.arange(len(intensity))
        for i in e:
            if intensity[i] == 0:
                continue
        
            alarm_start_time = (
                    start_time + np.random.exponential(1 / (strength * intensity[i])))
            if np.isnan(alarm_start_time):
                alarm_start_time = 100000000000000000
            c_2 = 0

            while True:

                if alarm_start_time > truncate or c_2 >= 1:
                    break

                # check if the alarm had happen
                same_alarm = self.current_observation[
                    (self.current_observation['Node'] == node) & (self.current_observation['Event'] == i)]
                same_alarm['End Time Stamp'] = same_alarm['Start Time Stamp'] + same_alarm['Last Time Stamp']
                same_alarm = same_alarm[same_alarm['End Time Stamp'] > self.current_time]
                if len(same_alarm) > self.limit_same_alarms:
                    break
                self.alarm_id += 1

                alarm_type = self.alarm_type_arr[node, i]
                alarm_last_time = np.max((10, np.random.exponential(10))) if alarm_type == 1 else 999999
                sub_alarm_list.append(
                    (self.alarm_id, i, alarm_start_time, alarm_last_time, flag,
                     father_id, father_node, father_event, alarm_type))

                c_2 += 1

                interval = np.random.exponential(1 / (strength * intensity[i]))

                alarm_start_time = alarm_start_time + interval

        return sub_alarm_list

    def _find_remove_index(self, node, event, alarm_table, alarm_id=-1, deleteById=False):

        def find_child_index(alarm_table, e_id):
            child_index = alarm_table[alarm_table['Father Id'] == e_id].index.tolist()
            descendants_index = [] + child_index
            for i in child_index:
                e = alarm_table.iloc[i, :]
                e_id = e['Alarm Id']

                grandchild_index = find_child_index(alarm_table, e_id)
                descendants_index += grandchild_index

            return descendants_index

        index = []
        if deleteById:
            index = alarm_table[alarm_table['Alarm Id'] == alarm_id].index.tolist()
        else:
            index = alarm_table[
                (alarm_table['Node'] == node) & (alarm_table['Event'] == event)].index.tolist()

        remove_index = [] + index
        
        for i in index:
            e = alarm_table.iloc[i, :]
            e_id = e['Alarm Id']
            remove_index += find_child_index(alarm_table, e_id)

        return remove_index

    def _get_state(self, alarm_data):
        state = np.zeros((self.node_num, self.event_num, 2))
        for row in alarm_data.iterrows():
            state[row[1]['Node'], row[1]['Event'], 0] = row[1]['Start Time Stamp']
            state[row[1]['Node'], row[1]['Event'], 1] += 1

        return state.flatten()

    def search_causal_order(self, matrix):
        """Obtain a causal order from the given matrix strictly.
        Parameters
        ----------
        matrix : array-like, shape (n_features, n_samples)
            Target matrix.
        Return
        ------
        causal_order : array, shape [n_features, ]
            A causal order of the given matrix on success, None otherwise.
        """
        causal_order = []

        row_num = matrix.shape[0]
        original_index = np.arange(row_num)

        while 0 < len(matrix):
            # find a row all of which elements are zero
            row_index_list = np.where(np.sum(np.abs(matrix), axis=1) == 0)[0]
            if len(row_index_list) == 0:
                break

            target_index = row_index_list[0]

            # append i to the end of the list
            causal_order.append(original_index[target_index])
            original_index = np.delete(original_index, target_index, axis=0)

            # remove the i-th row and the i-th column from matrix
            mask = np.delete(np.arange(len(matrix)), target_index, axis=0)
            matrix = matrix[mask][:, mask]

        if len(causal_order) != row_num:
            causal_order = None

        return causal_order

    def estimate_causal_order(self, matrix):
        """Obtain a lower triangular from the given matrix approximately.
        Parameters
        ----------
        matrix : array-like, shape (n_features, n_samples)
            Target matrix.
        Return
        ------
        causal_order : array, shape [n_features, ]
            A causal order of the given matrix on success, None otherwise.
        """
        causal_order = None

        # set the m(m + 1)/2 smallest(in absolute value) elements of the matrix to zero
        pos_list = np.argsort(np.abs(matrix), axis=None)
        pos_list = np.vstack(np.unravel_index(pos_list, matrix.shape)).T
        initial_zero_num = int(matrix.shape[0] * (matrix.shape[0] + 1) / 2)
        for i, j in pos_list[:initial_zero_num]:
            matrix[i, j] = 0

        for i, j in pos_list[initial_zero_num:]:
            # set the smallest(in absolute value) element to zero
            matrix[i, j] = 0

            causal_order = self.search_causal_order(matrix)
            if causal_order is not None:
                break

        return causal_order

    def seed(self, seed):
        np.random.seed(seed)
        random.seed(seed)