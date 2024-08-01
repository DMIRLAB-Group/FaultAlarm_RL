import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, Categorical

import numpy as np
import random

################################## PPO Policy ##################################
class RolloutBuffer:
	def __init__(self):
		self.actions = []
		self.states = []
		self.logprobs = []
		self.rewards = []
		self.is_terminals = []

	def add(self, state, action, reward, state_, done):
		self.rewards.append(reward)
		self.is_terminals.append(done)

	def clear(self):
		del self.actions[:]
		del self.states[:]
		del self.logprobs[:]
		del self.rewards[:]
		del self.is_terminals[:]


class ActorCritic(nn.Module):
	def __init__(self, state_dim, action_dim, hidden_units, has_continuous_action_space, action_std_init, device):
		super(ActorCritic, self).__init__()
		self.device = device
		self.has_continuous_action_space = has_continuous_action_space
		self.state_dim = state_dim
		self.action_dim = action_dim
		if has_continuous_action_space:
			self.action_dim = action_dim
			self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)
		# actor
		if has_continuous_action_space:
			self.actor = nn.Sequential(
				nn.Linear(state_dim, hidden_units),
				nn.Tanh(),
				nn.Linear(hidden_units, hidden_units),
				nn.Tanh(),
				nn.Linear(hidden_units, action_dim),
			)
		else:
			self.actor = nn.Sequential(
				nn.Linear(state_dim, hidden_units),
				nn.Tanh(),
				nn.Linear(hidden_units, hidden_units),
				nn.Tanh(),
				nn.Linear(hidden_units, action_dim),
				nn.Softmax(dim=-1)
			)
		# critic
		self.critic = nn.Sequential(
			nn.Linear(state_dim, hidden_units),
			nn.Tanh(),
			nn.Linear(hidden_units, hidden_units),
			nn.Tanh(),
			nn.Linear(hidden_units, 1)
		)

	def set_action_std(self, new_action_std):
		if self.has_continuous_action_space:
			self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(self.device)
		else:
			print("--------------------------------------------------------------------------------------------")
			print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
			print("--------------------------------------------------------------------------------------------")

	def forward(self):
		raise NotImplementedError

	def act(self, state):
		if self.has_continuous_action_space:
			action_mean = self.actor(state)
			cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
			dist = MultivariateNormal(action_mean, cov_mat)
		else:
			action_probs = self.actor(state)
			
			# add state mask ---------------------------------------------
			new_s = torch.reshape(state.detach(), (self.action_dim, 2))[:,1]
			state_mask = torch.where(new_s > 0, 1., 1e-8)
			action_probs = action_probs.mul(state_mask)
			# add state mask ---------------------------------------------

			dist = Categorical(action_probs)

		action = dist.sample()

		action_logprob = dist.log_prob(action)

		return action.detach(), action_logprob.detach()

	def evaluate(self, state, action):

		if self.has_continuous_action_space:
			action_mean = self.actor(state)
			
			action_var = self.action_var.expand_as(action_mean)
			cov_mat = torch.diag_embed(action_var).to(self.device)
			dist = MultivariateNormal(action_mean, cov_mat)

			# For Single Action Environments.
			if self.action_dim == 1:
				action = action.reshape(-1, self.action_dim)
		else:
			action_probs = self.actor(state)
			
			# add state mask ---------------------------------------------
			new_s = torch.reshape(state.detach(), (state.shape[0],self.action_dim, 2))[:,:,1]
			state_mask = torch.where(new_s > 0, 1., 1e-8)
			action_probs = action_probs.mul(state_mask)
			# add state mask ---------------------------------------------

			dist = Categorical(action_probs)
		action_logprobs = dist.log_prob(action)
		dist_entropy = dist.entropy()
		state_values = self.critic(state)

		return action_logprobs, state_values, dist_entropy


class PPO:
	def __init__(self, state_dim, action_dim, config, device):

		self.name = "PPO"
		self.device = device

		self.has_continuous_action_space = config["has_continuous_action_space"]
		self.hidden_units = config["hidden_units"]
		self.gamma = config["gamma"]
		self.eps_clip = config["eps_clip"]
		self.K_epochs = config["K_epochs"]
		self.lr_actor = config["lr_actor"]
		self.lr_critic = config["lr_critic"]
		self.normalize_advantage = config["normalize_advantage"]
		self.max_grad_norm = config["max_grad_norm"]
		self.batch_size = config["batch_size"]
		self.action_std = 0.6
		
		self.buffer = RolloutBuffer()

		self.policy = ActorCritic(state_dim, action_dim, self.hidden_units, self.has_continuous_action_space,
								self.action_std, device).to(device)
		self.optimizer = torch.optim.Adam([
			{'params': self.policy.actor.parameters(), 'lr': self.lr_actor},
			{'params': self.policy.critic.parameters(), 'lr': self.lr_critic}
		])

		self.policy_old = ActorCritic(state_dim, action_dim, self.hidden_units, self.has_continuous_action_space,
									self.action_std, device).to(device)
		self.policy_old.load_state_dict(self.policy.state_dict())

		self.MseLoss = nn.MSELoss()

	def set_action_std(self, new_action_std):
		if self.has_continuous_action_space:
			self.action_std = new_action_std
			self.policy.set_action_std(new_action_std)
			self.policy_old.set_action_std(new_action_std)
		else:
			print("--------------------------------------------------------------------------------------------")
			print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
			print("--------------------------------------------------------------------------------------------")

	def decay_action_std(self, action_std_decay_rate, min_action_std):
		print("--------------------------------------------------------------------------------------------")
		if self.has_continuous_action_space:
			self.action_std = self.action_std - action_std_decay_rate
			self.action_std = round(self.action_std, 4)
			if self.action_std <= min_action_std:
				self.action_std = min_action_std
				print("setting actor output action_std to min_action_std : ", self.action_std)
			else:
				print("setting actor output action_std to : ", self.action_std)
			self.set_action_std(self.action_std)

		else:
			print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")
		print("--------------------------------------------------------------------------------------------")

	def select_action(self, state):

		if self.has_continuous_action_space:
			with torch.no_grad():
				state = torch.FloatTensor(state).to(self.device)
				action, action_logprob = self.policy_old.act(state)

			self.buffer.states.append(state)
			self.buffer.actions.append(action)
			self.buffer.logprobs.append(action_logprob)

			return action.detach().cpu().numpy().flatten()
		else:
			with torch.no_grad():
				state = torch.FloatTensor(state).to(self.device)
				action, action_logprob = self.policy_old.act(state)

			self.buffer.states.append(state)
			self.buffer.actions.append(action)
			self.buffer.logprobs.append(action_logprob)

			return action.item()

	def update(self):
		# Update optimizer learning rate
        # self.update_learning_rate(self.policy.optimizer)

		loss_list = []
		pg_losses, value_losses ,entropy_losses = [], [], []

		# Optimize policy for K epochs
		for _ in range(self.K_epochs):
			batches = random.sample(range(len(self.buffer.states)), k=self.batch_size)
			# batches = range(len(self.buffer.states))
			
			# Monte Carlo estimate of returns
			rewards = []
			discounted_reward = 0
			bc_rewards = [ self.buffer.rewards[b] for b in batches ]
			bc_is_terminals = [ self.buffer.is_terminals[b] for b in batches ]
			for reward, is_terminal in zip(reversed(bc_rewards), reversed(bc_is_terminals)):
				if is_terminal:
					discounted_reward = 0

				discounted_reward = reward + (self.gamma * discounted_reward)
				rewards.insert(0, discounted_reward)

			# Normalizing the rewards
			rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
			# rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

			# convert list to tensor
			bc_states = [ self.buffer.states[b] for b in batches ]
			bc_actions = [ self.buffer.actions[b] for b in batches ]
			bc_logprobs = [ self.buffer.logprobs[b] for b in batches ]
			old_states = torch.squeeze(torch.stack(bc_states, dim=0)).detach().to(self.device)
			old_actions = torch.squeeze(torch.stack(bc_actions, dim=0)).detach().to(self.device)
			old_logprobs = torch.squeeze(torch.stack(bc_logprobs, dim=0)).detach().to(self.device)

		
			# Evaluating old actions and values
			logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

			# match state_values tensor dimensions with rewards tensor
			state_values = torch.squeeze(state_values)

			# Finding the ratio (pi_theta / pi_theta__old)
			ratios = torch.exp(logprobs - old_logprobs.detach())

			# Finding Surrogate Loss
			advantages = rewards - state_values.detach()

			# Normalization does not make sense if mini batchsize == 1
			if len(advantages) > 1:
				advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)


			surr1 = ratios * advantages
			surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

			policy_loss = torch.min(surr1, surr2).mean()
			value_loss = self.MseLoss(state_values, rewards)
			entropy_loss = -torch.mean(dist_entropy)
			
			pg_losses.append(policy_loss.item())
			value_losses.append(value_loss.item())
			entropy_losses.append(entropy_loss.item())

			# final loss of clipped objective PPO
			loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

			# self.log_loss.append(loss.mean().clone().data.cpu().numpy())
			loss_list.append(loss.mean().item())
			
			# take gradient step
			self.optimizer.zero_grad()
			loss.mean().backward()

			# Clip grad norm
			max_grad_norm = self.max_grad_norm
			torch.nn.utils.clip_grad_norm_(self.policy.actor.parameters(), max_grad_norm)
			torch.nn.utils.clip_grad_norm_(self.policy.critic.parameters(), max_grad_norm)

			self.optimizer.step()

		# Copy new weights into old policy
		self.policy_old.load_state_dict(self.policy.state_dict())

		# clear buffer
		self.buffer.clear()

		pg_loss_mean = np.mean(pg_losses)
		value_loss_mean = np.mean(value_losses)
		entropy_loss_mean = np.mean(entropy_losses)
		loss_mean = np.mean(loss_list)

		return loss_mean, pg_loss_mean, value_loss_mean, entropy_loss_mean

	def save(self, checkpoint_path):
		torch.save(self.policy_old.state_dict(), checkpoint_path)

	def load(self, checkpoint_path):
		self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
		self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))