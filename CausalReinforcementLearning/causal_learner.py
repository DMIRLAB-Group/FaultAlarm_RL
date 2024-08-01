import numpy as np
import pandas as pd
from tqdm import tqdm
import networkx as nx
from itertools import product
import pickle
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
import os
from graphviz import Digraph
import scipy.stats as stats


__MIN__ = -100000000000000

def check_and_create_dir(path: str):
    if os.path.exists(path):
        return
    os.mkdir(path)


def check_DAG(edge_mat):
    c_g = nx.from_numpy_array(edge_mat-np.diag(np.diag(edge_mat)), create_using = nx.DiGraph)
    return nx.is_directed_acyclic_graph(c_g)

class THP(object):

    def __init__(self, event_table: pd.DataFrame, decay, topo: nx.Graph = None, check_prior=lambda i: True,
                 init_structure: np.array = None, penalty='BIC', reg_parm=1,max_hop=5):
        """
        THP: a causal learning algorithm
        :param event_table: A pandas.DataFrame of events with columns  ['Node', 'Occurrence Time', 'Event Name']
        :param decay: The decay used in the exponential kernel
        :param topo: Topological graph of nodes
        :param check_prior: a prior function to check whether the adj satisfies the prior. (Input: adj_mat; Output: True or False)
        :param init_structure: adj of causal structure of prior knowledge
        :param penalty: 'BIC' or 'AIC' penalty
        :param max_hop: The maximum topological distance that the model considered
        """

        if (penalty not in {'BIC', 'AIC'}):
            raise Exception('Penalty is not supported')
        self.penalty = penalty

        self.reg_parm = reg_parm
        self.only_delete_edge = False

        self.node_list = np.array(topo.nodes)
        self.A = nx.adjacency_matrix(topo).todense().astype('float')
        self.D = np.mat(np.diag(np.array(self.A).sum(0))).astype('float')
        self.D_inv = np.mat(np.diag(1 / np.diagonal(self.D)))
        self.D_inv[self.D_inv == np.inf] = 0
        self.A_sys = self.D_inv * self.A * self.D_inv
        self.A_k = np.zeros([max_hop, *self.A.shape])
        self.A_sys_k = np.zeros([max_hop, *self.A.shape])
        for k in range(max_hop):
            self.A_k[k] = self.A ** k
            self.A_sys_k[k] = self.A_sys ** k

        self.event_table, self.event_names = self.get_event_table(event_table)

        self.node_names = self.event_table['Node'].unique()
        if (topo is None):
            self.topo = nx.Graph()
            self.topo.add_nodes_from(self.node_names)
            self.max_hop = 1
        else:
            self.topo = topo
            self.max_hop = max_hop

        self.decay = decay
        self.n = len(self.event_names)
        self.max_t = self.event_table['Start Time Stamp'].max()
        self.min_t = self.event_table['Start Time Stamp'].min()
        self.T = (self.max_t - self.min_t) * \
            len(self.event_table['Node'].unique())  # |V|x|T|

        if (init_structure is None):
            self.init_structure = np.eye(self.n, self.n)
        elif (not ((init_structure == 1) | (init_structure == 0)).all()):
            raise ValueError(
                'Elements of the adjacency matrix need to be 0 or 1')
        else:
            self.init_structure = np.array(init_structure)

            self.init_structure = self.init_structure - \
                np.diag(self.init_structure.diagonal()) + \
                np.eye(self.n, self.n)

        self.check_prior = check_prior
        if (not self.check_prior(self.init_structure)):
            raise ValueError('init structure is not satisfied with the prior')

        self.hist_likelihood = dict()
        for i in range(len(self.event_names)):
            self.hist_likelihood[i] = dict()

        self.event_table_groupby_NE = self.event_table.groupby('Node')
        self.decay_effect_without_end_time = self.get_decay_effect_integral_on_t()
        self.effect_tensor_decay_all = self.get_decay_effect_of_each_jump()

    def get_node_ind(self, node):
        return np.where(self.node_list == node)[0][0]

    def get_event_table(self, event_table: pd.DataFrame):
        event_table = event_table.copy()
        event_table.columns = ['Node', 'Start Time Stamp', 'Event Name']

        event_table['Times'] = np.zeros(len(event_table))
        event_table = event_table.groupby(
            ['Node', 'Start Time Stamp', 'Event Name']).count().reset_index()

        event_ind = event_table['Event Name'].astype('category')
        event_table['Event Ind'] = event_ind.cat.codes
        event_names = event_ind.cat.categories

        event_table.sort_values(['Node', 'Start Time Stamp', 'Event Ind'])
        return event_table, event_names

    def K_hop_neibors(self, node, K):
        if K == 0:
            return {node}
        else:
            return (set(nx.single_source_dijkstra_path_length(self.topo, node, K).keys()) - set(
                nx.single_source_dijkstra_path_length(self.topo, node, K - 1).keys()))

    def get_decay_effect_integral_on_t(self):
        decay_effect_without_end_time = np.zeros(
            [len(self.event_names), self.max_hop])
        for k in range(self.max_hop):
            decay_effect_without_end_time[:, k] = self.event_table.groupby('Event Ind').apply(lambda i: (
                (((1 - np.exp(-self.decay * (self.max_t - i['Start Time Stamp']))) / self.decay) * i['Times']) *
                i['Node'].apply(lambda j: self.A_sys_k[k, self.get_node_ind(j), :].sum())).sum())

        return decay_effect_without_end_time

    def get_decay_effect_of_each_jump(self):
        effect_tensor_decay_all = np.zeros(
            [self.max_hop, len(self.event_table), len(self.event_names)])
        for k in range(self.max_hop):
            event_table_array = self.event_table[[
                'Node', 'Start Time Stamp', 'Event Ind', 'Times']].values
            j = 0

            pre_effect = np.zeros(self.n)

            for item_ind in range(len(self.event_table)):

                node, start_t, ala_i, times = event_table_array[item_ind, [
                    0, 1, 2, 3]]
                last_node, last_start_t, last_ala_i, last_times = event_table_array[
                    item_ind - 1, [0, 1, 2, 3]]
                if ((last_node != node) or (last_start_t > start_t)):
                    j = 0
                    pre_effect = np.zeros(self.n)
                    try:
                        K_hop_neighbors_NE = self.K_hop_neibors(node, k)
                        neighbors_table = pd.concat(
                            [self.event_table_groupby_NE.get_group(i) for i in K_hop_neighbors_NE])
                        neighbors_table = neighbors_table.sort_values(
                            'Start Time Stamp')
                        neighbors_table_value = neighbors_table[
                            ['Node', 'Start Time Stamp', 'Event Ind', 'Times']].values
                    except ValueError as e:
                        K_hop_neighbors_NE = []
                if (len(K_hop_neighbors_NE) == 0):
                    continue

                cur_effect = pre_effect * \
                    np.exp((np.min((last_start_t - start_t, 0))) * self.decay)
                while (1):
                    try:
                        nei_node, nei_start_t, nei_ala_i, nei_times = neighbors_table_value[j, :]
                    except Exception as e:
                        break
                    if (nei_start_t < start_t):
                        cur_effect[int(nei_ala_i)] += nei_times * np.exp((nei_start_t - start_t) * self.decay) * \
                            self.A_sys_k[
                            k, self.get_node_ind(nei_node), self.get_node_ind(node)]

                        j += 1
                    else:
                        break
                pre_effect = cur_effect

                effect_tensor_decay_all[k, item_ind] = pre_effect
        return effect_tensor_decay_all

    def EM(self, edge_mat):
        '''
        :param edge_mat:    Adjacency matrix
        :return:            Return (likelihood, alpha matrix, mu vector)
        '''

        # if ((not check_DAG(edge_mat)) or (not self.check_prior(edge_mat))):
        #     return __MIN__, np.zeros([len(self.event_names), len(self.event_names)]), np.zeros(
        #         len(self.event_names))

        alpha = np.ones(
            [self.max_hop, len(self.event_names), len(self.event_names)])
        alpha = alpha * edge_mat
        mu = np.ones(len(self.event_names))
        L = 0

        for i in range(len(self.event_names)):

            Pa_i = set(np.where(edge_mat[:, i] == 1)[0])

            try:

                Li = self.hist_likelihood[i][tuple(Pa_i)][0]
                mu[i] = self.hist_likelihood[i][tuple(Pa_i)][2]
                for j in Pa_i:
                    alpha[:, j, i] = self.hist_likelihood[i][tuple(Pa_i)][1][j]
                L += Li
            except Exception as e:

                Li = __MIN__

                ind = np.where(self.event_table['Event Ind'] == i)
                X_i = self.event_table['Times'].values[ind]
                X_i_all = np.zeros_like(self.event_table['Times'].values)
                X_i_all[ind] = X_i
                while (1):

                    lambda_i_sum = (
                        self.decay_effect_without_end_time * alpha[:, :, i].T).sum() + mu[i] * self.T

                    lambda_for_i = np.zeros(len(self.event_table)) + mu[i]
                    for k in range(self.max_hop):
                        lambda_for_i += np.matmul(
                            self.effect_tensor_decay_all[k, :], alpha[k, :, i].T)
                    lambda_for_i = lambda_for_i[ind]

                    X_log_lambda = (X_i * np.log(lambda_for_i)).sum()

                    new_Li = -lambda_i_sum + X_log_lambda

                    decay = new_Li - Li

                    if (decay < 0.1):
                        Li = new_Li
                        L += Li
                        Pa_i_alpha = dict()
                        for j in Pa_i:
                            Pa_i_alpha[j] = alpha[:, j, i]
                        self.hist_likelihood[i][tuple(Pa_i)] = (
                            Li, Pa_i_alpha, mu[i])
                        break
                    Li = new_Li

                    # update mu
                    mu[i] = ((mu[i] / lambda_for_i) * X_i).sum() / self.T

                    # update alpha
                    for j in Pa_i:
                        for k in range(self.max_hop):
                            upper = ((alpha[k, j, i] * ((self.effect_tensor_decay_all)[k, :, j])[
                                ind] / lambda_for_i) * (X_i)).sum()
                            lower = self.decay_effect_without_end_time[j, k]
                            if (lower == 0):
                                alpha[k, j, i] = 0
                                continue
                            alpha[k, j, i] = upper / (lower)

                i += 1

        if (self.penalty == 'AIC'):
            return L - (len(self.event_names) + (edge_mat).sum()
                        * self.max_hop), alpha, mu
        if (self.penalty == 'BIC'):
            reg = (len(self.event_names) + (edge_mat).sum() * self.max_hop) * np.log(self.event_table['Times'].sum()) / 2
            reg *= self.reg_parm
            return L - reg, alpha, mu

    def EM_not_DAG(self, edge_mat):
        '''
        :param edge_mat:    Adjacency matrix
        :return:            Return (likelihood, alpha matrix, mu vector)
        '''

        alpha = np.ones(
            [self.max_hop, len(self.event_names), len(self.event_names)])
        alpha = alpha * edge_mat
        mu = np.ones(len(self.event_names))
        L = 0

        for i in range(len(self.event_names)):

            Pa_i = set(np.where(edge_mat[:, i] == 1)[0])

            try:

                Li = self.hist_likelihood[i][tuple(Pa_i)][0]
                mu[i] = self.hist_likelihood[i][tuple(Pa_i)][2]
                for j in Pa_i:
                    alpha[:, j, i] = self.hist_likelihood[i][tuple(Pa_i)][1][j]
                L += Li
            except Exception as e:

                Li = __MIN__

                ind = np.where(self.event_table['Event Ind'] == i)
                X_i = self.event_table['Times'].values[ind]
                X_i_all = np.zeros_like(self.event_table['Times'].values)
                X_i_all[ind] = X_i
                while (1):

                    # Calculate the first part of the likelihood
                    lambda_i_sum = (
                        self.decay_effect_without_end_time * alpha[:, :, i].T).sum() + mu[i] * self.T

                    # Calculate the second part of the likelihood
                    lambda_for_i = np.zeros(len(self.event_table)) + mu[i]
                    for k in range(self.max_hop):
                        lambda_for_i += np.matmul(
                            self.effect_tensor_decay_all[k, :], alpha[k, :, i].T)
                    lambda_for_i = lambda_for_i[ind]

                    X_log_lambda = (X_i * np.log(lambda_for_i)).sum()

                    new_Li = -lambda_i_sum + X_log_lambda

                    # Iteration termination condition

                    decay = new_Li - Li

                    if (decay < 0.1):
                        Li = new_Li
                        L += Li
                        Pa_i_alpha = dict()
                        for j in Pa_i:
                            Pa_i_alpha[j] = alpha[:, j, i]
                        self.hist_likelihood[i][tuple(Pa_i)] = (
                            Li, Pa_i_alpha, mu[i])
                        break
                    Li = new_Li

                    # update mu
                    mu[i] = ((mu[i] / lambda_for_i) * X_i).sum() / self.T

                    # update alpha
                    for j in Pa_i:
                        for k in range(self.max_hop):
                            upper = ((alpha[k, j, i] * ((self.effect_tensor_decay_all)[k, :, j])[
                                ind] / lambda_for_i) * (X_i)).sum()
                            lower = self.decay_effect_without_end_time[j, k]
                            if (lower == 0):
                                alpha[k, j, i] = 0
                                continue
                            lapace_para = 1
                            alpha[k, j, i] = upper / \
                                (lower + lapace_para)  # equal to L1

                i += 1

        if (self.penalty == 'AIC'):
            return L - (len(self.event_names) + (edge_mat).sum()
                        * self.max_hop), alpha, mu
        if (self.penalty == 'BIC'):
            return L - (len(self.event_names) + (edge_mat).sum() * self.max_hop) * np.log(
                self.event_table['Times'].sum()) / 2, alpha, mu

    def one_step_change_iterator(self, edge_mat):
        return map(lambda e: self.one_step_change(edge_mat, e),
                   product(range(len(self.event_names)), range(len(self.event_names))))

    def one_step_change(self, edge_mat, e):
        j, i = e
        if (j == i):
            return edge_mat
        new_edge_mat = edge_mat.copy()

        if (new_edge_mat[j, i] == 1):
            new_edge_mat[j, i] = 0
            return new_edge_mat
        else:
            if self.only_delete_edge:
                return new_edge_mat
            new_edge_mat[j, i] = 1
            new_edge_mat[i, j] = 0
            return new_edge_mat

    def Hill_Climb(self, edge_mat=None):
        if edge_mat is None:
            edge_mat = self.init_structure - \
                np.diag(self.init_structure.diagonal()) + np.eye(self.n, self.n)

        result = self.EM(edge_mat)
        L = result[0]
        while (1):
            stop_tag = True
            for new_edge_mat in (
                    list(self.one_step_change_iterator(edge_mat))):
                new_result = self.EM(new_edge_mat)
                new_L = new_result[0]

                if (new_L > L):
                    result = new_result
                    L = new_L
                    stop_tag = False
                    edge_mat = new_edge_mat
            if (stop_tag):
                return result, edge_mat


class CausalLearner:
    def __init__(self, node_num, event_num, subset_size, dataset_path,random_g,seed,max_hop=2):
        self.subset_size = subset_size
   
        self.dataset_path = dataset_path
        self.node_num = node_num
        self.event_num = event_num
        self.seed = seed
        self.history_events, self.topology, self.env_graph, self.env_edge_mat, self.env_mu,\
        self.env_alpha, self.true_causal_order = pickle.load(open(dataset_path, 'rb'))

        self.event_order = []

        self.max_hop = max_hop
        self.thp_decay = 0.00005
        self.random_g = random_g
        A = nx.adjacency_matrix(self.topology).todense().astype('float')
        D = np.mat(np.diag(np.array(A).sum(0))).astype('float')
        D_inv = np.mat(np.diag(1 / np.diagonal(D)))
        D_inv[D_inv == np.inf] = 0
        A_sys = D_inv * A * D_inv
        A_sys_k = np.zeros([self.max_hop, *A.shape])
        A_k = np.zeros([self.max_hop, *A.shape])
        for k in range(self.max_hop):
            A_k[k] = A ** k
            A_sys_k[k] = A_sys ** k
        self.A_sys_k = A_sys_k
    
    def learn_causal_graph(self,reg_parm=0.2):
        # topology
        topology_mat = nx.adjacency_matrix(self.topology).todense().astype('float')
        causal_info = {
            "subset_size": self.subset_size,
            "max_hop": self.max_hop
        }

        # learn causal graph by THP
        event_table = self.history_events[['Node', 'Start Time Stamp', 'Event']]
        
        THP_para = {'decay': self.thp_decay, 'max_hop': self.max_hop, 'reg_parm': reg_parm}

        self.event_table = event_table
        self.THP_para = THP_para
        print('---------------------------begin causal learning ------------------------------------')
        thp = THP(event_table, topo=self.topology, **THP_para)
        [likelihood, alpha_mat, mu], edge_mat = thp.Hill_Climb()
        for i in range(len(edge_mat)):
            edge_mat[i, i] = 0
        # print performance
        recall, precision, f1 = self.get_performance(edge_mat, self.env_edge_mat, 0.000)

        # learn causal order
        causal_order = self.estimate_causal_order(edge_mat.copy())
        causal_order = np.flipud(causal_order)
        self.event_order = causal_order

        print('---------------------------finish causal learning ------------------------------------')
        
        causal_info["alpha_mat"] = alpha_mat
        causal_info["mu"] = mu
        causal_info["event_order"] = causal_order
        causal_info["edge_mat"] = edge_mat
        causal_info['recall'] = recall
        causal_info['precision'] = precision
        causal_info['f1'] = f1
        self.alpha = alpha_mat
        
        if self.random_g == 1 :
            model = pickle.load(open(f'faultAlarm/data/random_edgemat/random_model_seed_{self.seed}.pkl', 'rb'))
            causal_info["event_order"] = model['random_causal_order']
            causal_info["edge_mat"] = model['random_edge_mat']
            causal_info['recall'] = model['random_recall']
            causal_info['precision'] = model['random_precision']
            causal_info['f1'] = model['random_f1']
            
        return causal_info

    def get_performance(self, adj, true_adj, threshold=0.0):
        
        precision = metrics.precision_score(
            true_adj.ravel(), adj.ravel() > threshold)
        recall = metrics.recall_score(true_adj.ravel(), adj.ravel() > threshold)
        f1 = metrics.f1_score(true_adj.ravel(), adj.ravel() > threshold)
        # print(f' precision:{precision}  ,  recall:{recall}  ,  f1:{f1}')
        return recall, precision, f1

    def update_edge_mat(self, new_edge_mat, reg_parm=0.25):
        
        thp = THP(self.event_table, topo=self.topology, **self.THP_para)
        thp.only_delete_edge = True
        thp.reg_parm = reg_parm
        [likelihood, alpha_mat, mu], edge_mat = thp.Hill_Climb(new_edge_mat)
        for i in range(len(edge_mat)):
            edge_mat[i, i] = 0
        # print performance
        recall, precision, f1 = self.get_performance(edge_mat, self.env_edge_mat, 0.000)
        
        return edge_mat, alpha_mat, f1, precision, recall

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