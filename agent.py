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
        self.n_epoches = self.n_epoches
        self.gae_lambda = gae_lambda

        self.actor = actor(n_actions, input_dims, alpha)
        self.critic = critic(input_dims, alpha)

    def store_memory(self, states, probs, actions, vals, reward, done):
        PPOMemory.store_memory(states, probs, actions, vals, reward, done)

    def save_models(self):
        print('...........saving models...........')
        actor.save_checkpoint()
        critic.save_checkpoint()


    def load_models(self):
        print('...........loading models...........')
        actor.load_checkpoint()
        critic.load_checkpoint()

    def choose_action(self, observation):
        state = torch