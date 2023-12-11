import gym
import pygame
import numpy as np
import sys


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

env = CustomEnv()

pygame.init()
screen_size = 400
screen = pygame.display.set_mode((screen_size, screen_size))
clock = pygame.time.Clock()

num_states = env.observation_space.shape[0]
num_actions = env.action_space.n
q_values = np.zeros((10, 10, num_actions))


learning_rate = 0.1
gamma = 0.9
epsilon = 1

def egreedy_policy(q_values, state, epsilon):
    # Получение случайного числа из равномерного распределения между 0 и 1,
    # если число меньше epsilon, выбираем случайное действие
    if np.random.random() < epsilon:
        return np.random.choice(4)
    # Иначе выбираем действие с наибольшим значением
    else:
        return np.argmax(q_values[state])

for _ in range(500):
    state = env.reset()
    done = False

    # Пока эпизод не завершен
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Выбираем действие
        action = egreedy_policy(q_values, state, epsilon)

        # Выполняем действие
        next_state, reward, done, _ = env.step(action)

        # Обновляем значения q_values
        td_target = reward + gamma * np.max(q_values[next_state[0], next_state[1]])
        td_error = td_target - q_values[state[0], state[1], action]
        q_values[state[0], state[1], action] += learning_rate * td_error

        # Обновляем состояние
        state = next_state

        img = pygame.Surface((screen_size, screen_size))
        img.fill((255, 255, 255))
        agent_pos = state
        goal_pos = env.goal_position
        obstacle_pos = env.obstacle_position
        pygame.draw.rect(img, (128, 128, 128), (
        obstacle_pos[1] * int(screen_size / 10), obstacle_pos[0] * int(screen_size / 10), int(screen_size / 10),
        int(screen_size / 10)))

        pygame.draw.rect(img, (0, 0, 255), (agent_pos[1] * int(screen_size / 10), agent_pos[0] * int(screen_size / 10), int(screen_size / 10), int(screen_size / 10)))
        pygame.draw.rect(img, (0, 255, 0), (goal_pos[1] * int(screen_size / 10), goal_pos[0] * int(screen_size / 10), int(screen_size / 10), int(screen_size / 10)))
        screen.blit(img, (0, 0))
        pygame.display.flip()
        clock.tick(100)
pygame.quit()

