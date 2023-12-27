import gym
import pygame
import numpy as np
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as distributions
import torch.nn.functional as F

# Определение среды
class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        self.action_space = gym.spaces.Discrete(4)  # Возможные действия: вверх, вниз, влево, вправо
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(2,), dtype=np.uint8)

        self.agent_position = [0, 0]
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

    def reset(self):
        self.agent_position = [0, 0]
        self.obstacle_position = [4, 4]  # Обнуление позиции препятствия
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
    old_values = torch.stack(old_values).squeeze(-1)
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


total_rewards = []
for epoch in range(num_epochs):
    states = []
    actions = []
    log_probs = []
    values = []
    rewards = []
    state = env.reset()

    epoch_reward = 0

    state = env.reset()
    for step in range(num_steps):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        state_tensor = torch.from_numpy(state.astype(np.float32))
        logits, value = policy(state_tensor)

        dist = distributions.Categorical(logits=logits)
        action = dist.sample()

        next_state, reward, done, _ = env.step(action.item())
        log_prob = dist.log_prob(action)
        states.append(state_tensor)
        actions.append(action)
        log_probs.append(log_prob)
        values.append(value)
        rewards.append(torch.tensor([reward], dtype=torch.float32))

        state = next_state

        if done:
            break

    next_state_tensor = torch.from_numpy(next_state.astype(np.float32))
    _, next_value = policy(next_state_tensor)
    rewards.append(next_value.detach())
    total_rewards.append(sum(rewards))
    ppo_update(policy, optimizer, states, actions, log_probs, values, rewards[:-1], gamma, eps_clip)
    if epoch % 10 == 0:
        print('Epoch {} - Average Reward: {}'.format(epoch, np.mean(total_rewards[-10:])))

    img = pygame.Surface((screen_size, screen_size))
    img.fill((255, 255, 255))
    agent_pos = state
    goal_pos = env.goal_position
    obstacle_pos = env.obstacle_position
    pygame.draw.rect(img, (128, 128, 128), (
        obstacle_pos[1] * int(screen_size / 10), obstacle_pos[0] * int(screen_size / 10), int(screen_size / 10),
        int(screen_size / 10)))

    pygame.draw.rect(img, (0, 0, 255), (
        agent_pos[1] * int(screen_size / 10), agent_pos[0] * int(screen_size / 10), int(screen_size / 10),
        int(screen_size / 10)))
    pygame.draw.rect(img, (0, 255, 0), (
        goal_pos[1] * int(screen_size / 10), goal_pos[0] * int(screen_size / 10), int(screen_size / 10),
        int(screen_size / 10)))
    screen.blit(img, (0, 0))
    pygame.display.flip()
    clock.tick(100)
pygame.quit()