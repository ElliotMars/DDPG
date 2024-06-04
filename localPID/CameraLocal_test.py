import gym
import torch
import torch.nn as nn
import os
import pygame
import numpy as np

from CameraLocal_agent import STEP_LEN

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", device)

env = gym.make(id='CameraControlLocal-v0')

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=ANN):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)


    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        # 将 x 的每个元素与对应的缩放因子相乘
        scaled_x = torch.zeros_like(x)
        for i in range(len(STEP_LEN)):
            scaled_x[:, i] = x[:, i] * STEP_LEN[i]
        x = scaled_x
        return x

# Test phase
current_path = os.path.dirname(os.path.realpath(__file__))
model = current_path + '/train/'
actor_path = model + 'ddpg_actor_20240527205014.pth'

actor = Actor().to(device)
actor.load_state_dict(torch.load(actor_path))

num_episodes = 100
for episode_i in range(num_episodes):
    state, others = env.reset()
    episode_reward = 0
    done = False
    count = 0

    for step_i in range(200):
        action = actor(torch.FloatTensor(state).unsqueeze(0).to(device)).detach().cpu().numpy()[0]
        next_state, reward, done, truncation, _ = env.step(action)
        episode_reward += reward
        state = next_state
        count += 1
        print(f"{count}:{action}")


#pygame.quit()
env.close()