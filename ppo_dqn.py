import gym
import pygame
import numpy as np
import sys
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as distributions
import torch.nn.functional as F
from collections import deque

# Определение среды
class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        self.action_space = gym.spaces.Discrete(4)  # Возможные действия: вверх, вниз, влево, вправо
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(2,), dtype=np.uint8)

        self.agent_position = [9, 0]
        self.goal_position = [9, 9]
        self.obstacle_position = [4, 4]

        self.max_steps = 100
        self.current_step = 0

        self.reset()

    def step(self, action):
        self.current_step += 1

        if action == 0:  # Вверх
            new_position = [max(0, self.agent_position[0] - 1), self.agent_position[1]]
        elif action == 1:  # Вниз
            new_position = [min(9, self.agent_position[0] + 1), self.agent_position[1]]
        elif action == 2:  # Влево
            new_position = [self.agent_position[0], max(0, self.agent_position[1] - 1)]
        elif action == 3:  # Вправо
            new_position = [self.agent_position[0], min(9, self.agent_position[1] + 1)]

        if new_position != self.obstacle_position:  # Проверка на попадание на позицию препятствия
            self.agent_position = new_position

        # Вычисление вознаграждения и флага окончания
        if self.agent_position == self.goal_position:
            reward = 1
            done = True
        else:
            reward = 0
            done = self.current_step >= self.max_steps

        # Возвращение нового состояния, вознаграждения и флага окончания
        return self._get_obs(), reward, done, {}

    def reset(self, agent_id=None):
        if agent_id == 'ppo':
            self.agent_position = [0, 0]  # начальное состояние для PPO агента
        elif agent_id == 'dqn':
            self.agent_position = [9, 0]  # начальное состояние для DQN агента
        else:
            self.agent_position = [0, 0]
        self.obstacle_position = [4, 4]
        self.current_step = 0
        return self._get_obs()

    def _get_obs(self):
        if self.agent_position == self.obstacle_position:
            return np.array([255, 0])  # Агент на позиции препятствия
        else:
            return np.array(self.agent_position)


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()

        self.fc1 = nn.Linear(2, 32)
        self.fc2 = nn.Linear(32, 32)
        self.action_head = nn.Linear(32, 4)  # Голова политики для действий
        self.value_head = nn.Linear(32, 1)  # Голова оценщика для значения

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        action_probs = self.action_head(x)
        state_values = self.value_head(x)
        return action_probs, state_values

env = CustomEnv()
pygame.init()
screen_size = 400
screen = pygame.display.set_mode((screen_size, screen_size))
clock = pygame.time.Clock()

policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=0.01)
num_epochs = 500
num_steps = 100
gamma = 0.2
eps_clip = 0.9

def ppo_update(policy, optimizer, states, actions, old_log_probs, old_values, rewards, gamma, eps_clip):
    # Конвертация списков в тензоры
    states = torch.stack(states)
    actions = torch.stack(actions)
    old_log_probs = torch.stack(old_log_probs)
    old_values = old_values.clone().detach()
    returns = calculate_returns(rewards, gamma)
    advantages = returns - old_values

    action_probs, state_values = policy(states)
    dist = distributions.Categorical(logits=action_probs)
    new_log_probs = dist.log_prob(actions)

    # Вычисляем отношения вероятностей
    ratios = torch.exp(new_log_probs - old_log_probs.detach())

    # Вычисляем PPO-объектив с использованием clip-функции
    surr1 = ratios * advantages.detach()
    surr2 = torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * advantages.detach()
    # Actor loss
    policy_loss = -torch.min(surr1, surr2).mean()

    # Critic loss
    value_loss = F.mse_loss(state_values.squeeze(), returns.detach())


    # Общий loss
    loss = policy_loss + 0.5 * value_loss

    # Выполнить шаг оптимизации
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def calculate_returns(rewards, gamma):
    R = 0
    returns = []
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return torch.tensor(returns)
def calculate_advantages(returns, values):
    advantages = returns - values
    if len(advantages) > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    else:
        advantages = advantages - advantages.mean()

    return advantages


def calculate_policy_loss(log_probs, advantages, eps_clip):
    ratio = torch.exp(log_probs - log_probs.detach())
    surrogate1 = ratio * advantages
    surrogate2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantages
    policy_loss = -torch.min(surrogate1, surrogate2).mean()
    return policy_loss


def calculate_value_loss(values, returns):
    return F.mse_loss(values, returns.detach())


def compute_returns(rewards, gamma):
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return returns


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        return self.fc2(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.array(state), np.array(action), np.array(reward, dtype=np.float32), np.array(next_state), np.array(
            done)

    def __len__(self):
        return len(self.buffer)


def optimize_dqn(dqn_model, target_dqn_model, replay_buffer, optimizer, batch_size, gamma):
    if len(replay_buffer) < batch_size:
        return

    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    state = torch.FloatTensor(state)
    next_state = torch.FloatTensor(next_state)
    action = torch.LongTensor(action)
    reward = torch.FloatTensor(reward)
    done = torch.FloatTensor(done)

    q_values = dqn_model(state)
    next_q_values = target_dqn_model(next_state)
    q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    next_q_value = next_q_values.max(1)[0]
    expected_q_value = reward + gamma * next_q_value * (1 - done)

    loss = F.mse_loss(q_value, expected_q_value.detach())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def update_target(current_model, target_model):
    target_model.load_state_dict(current_model.state_dict())

dqn = DQN(env.observation_space.shape[0], env.action_space.n)
target_dqn = DQN(env.observation_space.shape[0], env.action_space.n)
dqn_optimizer = optim.Adam(dqn.parameters())
replay_buffer = ReplayBuffer(10000)
update_target(dqn, target_dqn)
total_rewards = []
policy = Policy()
ppo_optimizer = optim.Adam(policy.parameters(), lr=0.01)
replay_buffer = ReplayBuffer(1000)
batch_size = 64
gamma = 0.99
target_update_frequency = 50

dqn_rewards = []  # Список наград для DQN агента
for epoch in range(num_epochs):
    ppo_state = env.reset(agent_id='ppo')
    dqn_state = env.reset(agent_id='dqn')
    ppo_epoch_rewards = []
    dqn_epoch_rewards = []
    ppo_states = []
    ppo_actions = []
    ppo_log_probs = []
    ppo_values = []
    ppo_rewards = []

    for step in range(num_steps):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        ppo_state_tensor = torch.from_numpy(np.array(ppo_state, dtype=np.float32))
        logits, value = policy(ppo_state_tensor)
        dist = distributions.Categorical(logits=logits)
        ppo_action = dist.sample()

        ppo_next_state, ppo_reward, ppo_done, _ = env.step(ppo_action.item())
        ppo_epoch_rewards.append(ppo_reward)

        ppo_log_probs.append(dist.log_prob(ppo_action))
        ppo_values.append(value.item())
        ppo_states.append(ppo_state_tensor)
        ppo_actions.append(ppo_action)

        ppo_state = ppo_next_state

        dqn_state_tensor = torch.from_numpy(dqn_state).float().unsqueeze(0)
        with torch.no_grad():
            dqn_action_values = dqn(dqn_state_tensor)
        dqn_action = dqn_action_values.max(1)[1].view(1, 1).item()

        dqn_next_state, dqn_reward, dqn_done, _ = env.step(dqn_action)


        replay_buffer.push(dqn_state, dqn_action, dqn_reward, dqn_next_state, int(dqn_done))
        dqn_state = dqn_next_state


        if ppo_done or dqn_done:
            break

    if len(ppo_states) > 0:
        ppo_returns = calculate_returns(ppo_epoch_rewards, gamma)
        ppo_advantages = calculate_advantages(ppo_returns, torch.tensor(ppo_values))
        ppo_update(policy, ppo_optimizer, ppo_states, ppo_actions, ppo_log_probs,
                   torch.tensor(ppo_values), ppo_returns, gamma, eps_clip)

    if len(replay_buffer) > batch_size:
        optimize_dqn(dqn, target_dqn, replay_buffer, dqn_optimizer, batch_size, gamma)

    if epoch % target_update_frequency == 0:
        update_target(dqn, target_dqn)


    img = pygame.Surface((screen_size, screen_size))
    img.fill((255, 255, 255))
    ppo_pos = ppo_state
    dqn_pos = dqn_state
    goal_pos = env.goal_position
    obstacle_pos = env.obstacle_position


    def draw_environment(agent_ppo_pos, agent_dqn_pos, goal_pos, obstacle_pos):
        img.fill((255, 255, 255))
        pygame.draw.rect(img, (128, 128, 128), (
            obstacle_pos[1] * int(screen_size / 10), obstacle_pos[0] * int(screen_size / 10),
            int(screen_size / 10), int(screen_size / 10)
        ))
        pygame.draw.rect(img, (0, 0, 255), (
            agent_ppo_pos[1] * int(screen_size / 10), agent_ppo_pos[0] * int(screen_size / 10),
            int(screen_size / 10), int(screen_size / 10)
        ))
        pygame.draw.rect(img, (255, 0, 0), (
            agent_dqn_pos[1] * int(screen_size / 10), agent_dqn_pos[0] * int(screen_size / 10),
            int(screen_size / 10), int(screen_size / 10)
        ))
        pygame.draw.rect(img, (0, 255, 0), (
            goal_pos[1] * int(screen_size / 10), goal_pos[0] * int(screen_size / 10),
            int(screen_size / 10), int(screen_size / 10)
        ))

        screen.blit(img, (0, 0))
        pygame.display.flip()
    draw_environment(ppo_pos, dqn_pos, env.goal_position, env.obstacle_position)
    clock.tick(60)
pygame.quit()
