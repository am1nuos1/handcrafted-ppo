import os 
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from actor import ActorNetwork as actor
from critic import CriticNetwork as critic
from memory import PPOMemory


class Agent:
    def __init__(self, n_actions, input_dims, gamma = 0.99, alpha = 0.0003,
                 gae_lambda = 0.95, policy_clip = 0.2, batch_size = 64,
                 N = 2048, n_epochs = 10):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda

        self.actor = actor(n_actions, input_dims, alpha)
        self.critic = critic(input_dims, alpha)
        self.memory = PPOMemory(batch_size)

    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def store_memory(self, state, action, probs, vals, reward, done):
        self.remember(state, action, probs, vals, reward, done)

    def save_models(self):
        print('...........saving models...........')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()


    def load_models(self):
        print('...........loading models...........')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def choose_action(self, observation):
        state = torch.tensor(
            observation, dtype=torch.float32, device=self.actor.device
        ).flatten().unsqueeze(0)
        dist = self.actor(state)
        value = self.critic(state)
        action = dist.sample()
        
        probs = torch.squeeze(dist.log_prob(action)).item()
        action = torch.squeeze(action).item()
        value = torch.squeeze(value).item()

        return action, probs, value
    
    def learn(self):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr,\
            reward_arr, dones_arr, batches = \
                    self.memory.generate_batches()
            
            values = vals_arr
            advantages = np.zeros(len(reward_arr), dtype= np.float32)

            for t in range(len(reward_arr) -1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr)-1):
                    a_t += discount*(reward_arr[k] + self.gamma*values[k+1]*\
                                     (1- int(dones_arr[k])) - values[k])
                    discount *= self.gamma * self.gae_lambda
                advantages[t] = a_t

            advantages = torch.tensor(advantages).to(self.actor.device)
            values = torch.tensor(values).to(self.actor.device)

            for batch in batches:
                states = torch.tensor(
                    state_arr[batch], dtype=torch.float32, device=self.actor.device
                )
                old_probs = torch.tensor(old_prob_arr[batch]).to(self.actor.device)
                actions = torch.tensor(action_arr[batch]).to(self.actor.device)

                dist = self.actor(states)
                critic_value = self.critic(states)

                critic_value = torch.squeeze(critic_value)
                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()
                weighted_probs = advantages[batch] * prob_ratio
                weighted_clipped_probs = torch.clamp(
                    prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip
                ) * advantages[batch]
                actor_loss = -torch.min(weighted_clipped_probs, weighted_probs).mean()
                
                returns = advantages[batch] + values[batch]
                critic_loss = (returns - critic_value) **2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5*critic_loss
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()
                
        self.memory.clear_memory()
