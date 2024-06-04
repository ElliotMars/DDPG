import gym
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import time
from DC_agent import DCAgent, ANGLEX0, ANGLEY0, STEP_LEN
import random
import my_env

# Initialize environment
env = gym.make(id='DirectControl-v0')
STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.shape[0]

# Hyperparameters
NUM_EPISODE = 500
#NUM_PHASE = 6 #需要被600整除，同时需要修改 self.iterations
NUM_STEP = 600
EPSILON_START = 1.0
EPSILON_END = 0.02
EPSILON_DECAY = 0.5 * NUM_EPISODE * NUM_STEP

class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=30, theta=0.01, dt=0.05):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt

    def noise(self, anglex, angley):
        self.x_prev = np.array([anglex, angley])
        dx = self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.random.normal(size=self.mu.shape) * np.sqrt(self.dt)
        #dx[2] *= (STEP_LEN[2] / STEP_LEN[0])
        for i in range(len(dx)):
            dx[i] = min(max(dx[i], -STEP_LEN[i]), STEP_LEN[i])
        return dx

# Initialize agent
agent = DCAgent(STATE_DIM, ACTION_DIM)
OUActionNoise = OrnsteinUhlenbeckActionNoise(mu=np.array([ANGLEX0, ANGLEY0]))

# Training Loop
REWARD_BUFFER = np.empty(shape=NUM_EPISODE)
# ANGLE_BUFFER = np.empty(shape=(NUM_EPISODE,2))
for episode_i in range(NUM_EPISODE):
    state = env.reset()  # state: ndarray, others: dict
    episode_reward = 0

    for step_i in range(NUM_STEP):
        # Select action
        epsilon = np.interp(x=episode_i * NUM_STEP + step_i, xp=[0, EPSILON_DECAY],
                            fp=[EPSILON_START, EPSILON_END])  # interpolation
        random_sample = random.random()
        if random_sample <= epsilon:
            action = OUActionNoise.noise(env.anglex, env.angley)

            # for i in range(ACTION_DIM):
            #     action = np.empty(shape=(3,))
            #     action[i] = np.random.uniform(low=-STEP_LEN[i], high=STEP_LEN[i])
        else:
            action = agent.get_action(state)
        note_action = action / STEP_LEN#1
        # Execute action at and observe reward rt and observe new state st+1
        #print('STEP: ', step_i + 1)
        next_state, reward, done, truncation, info = env.step(action)
        # Store transition (st; at; rt; st+1) in R
        agent.replay_buffer.add_memo(state, note_action, reward, next_state, done)

        state = next_state
        episode_reward += reward

        agent.update()
        if done:
            break

    REWARD_BUFFER[episode_i] = episode_reward
    # ANGLE_BUFFER[episode_i][0] = env.anglex
    # ANGLE_BUFFER[episode_i][1] = env.angley
    print(f"Episode: {episode_i + 1}, Reward: {round(episode_reward, 2)}")

current_path = os.path.dirname(os.path.realpath(__file__))
train = current_path + '/train/'
timestamp = time.strftime("%Y%m%d%H%M%S")

# Save models
torch.save(agent.actor.state_dict(), train + f'ddpg_actor_{timestamp}.pth')
torch.save(agent.critic.state_dict(), train + f'ddpg_critic_{timestamp}.pth')

# max_index = np.argmax(REWARD_BUFFER)
# print('Final: ', 'Kp = ', env.Kp, ', Ki = ', env.Ki, ', Kd = ', env.Kd, 'Reward = ', REWARD_BUFFER[-1])
# print('Best: ', 'Kp = ', PID_BUFFER[max_index][0], ', Ki = ', PID_BUFFER[max_index][1],
#       ', Kd = ', PID_BUFFER[max_index][2], 'Reward = ', REWARD_BUFFER[max_index])
#
# with open('train/output_' + f'{timestamp}.txt', 'w') as f:
#     # 写入内容到文件中
#     f.write(f'Final: Kp = {env.Kp}, Ki = {env.Ki}, Kd = {env.Kd}, Reward = {REWARD_BUFFER[-1]}\n')
#     f.write(f'Best: Kp = {PID_BUFFER[max_index][0]}, Ki = {PID_BUFFER[max_index][1]}, Kd = {PID_BUFFER[max_index][2]},'
#             f' Reward = {REWARD_BUFFER[max_index]}\n')
#     f.write(f'NUM_EPISODE = {NUM_EPISODE}, NUM_PHASE = {NUM_PHASE}, NUM_STEP = {NUM_STEP}\n'
#             f'EPISILON_START = {EPSILON_START}, EPSILON_END = {EPSILON_END}, EPSILON_DECAY = {EPSILON_DECAY}\n')


# Close environment
env.close()

# Save the rewards as txt file
# np.savetxt(train + f'/ddpg_reward_{timestamp}.txt', REWARD_BUFFER)
np.savetxt(train + f'/ddpg_reward_{timestamp}.csv', REWARD_BUFFER, delimiter=',')

# Plot rewards using ax.plot()
plt.plot(REWARD_BUFFER)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('DDPG Reward')
plt.grid()
plt.savefig(train + f'/ddpg_reward_{timestamp}.png')
plt.show()


