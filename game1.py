import pygame
import sys
import random
import numpy as np
from pygame.math import Vector2

# Constants
size = 30
number = 20
SCREEN_SIZE = number * size
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
FPS = 20  

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
pygame.display.set_caption('Snake Game using AI')
clock = pygame.time.Clock()
font = pygame.font.Font(None, 36)

# Directions
UP = Vector2(0, -1)
DOWN = Vector2(0, 1)
LEFT = Vector2(-1, 0)
RIGHT = Vector2(1, 0)
DIRECTIONS = [UP, DOWN, LEFT, RIGHT]

class SnakeGame:
    def __init__(self):
        self.reset()

    def reset(self):
        self.snake_body = [Vector2(number // 2, number // 2)]
        self.direction = random.choice(DIRECTIONS)
        self.place_fruit()
        self.score = 0
        self.frame_iteration = 0
        return self.get_state()

    def place_fruit(self):
        self.fruit = Vector2(random.randint(0, number - 1), random.randint(0, number - 1))
        while self.fruit in self.snake_body:
            self.fruit = Vector2(random.randint(0, number - 1), random.randint(0, number - 1))

    def step(self, action):
        self.frame_iteration += 1

        # Enforce movement constraints
        if action == 0 and self.direction != DOWN:  # UP
            self.direction = UP
        elif action == 1 and self.direction != UP:  # DOWN
            self.direction = DOWN
        elif action == 2 and self.direction != RIGHT:  # LEFT
            self.direction = LEFT
        elif action == 3 and self.direction != LEFT:  # RIGHT
            self.direction = RIGHT

        new_head = self.snake_body[0] + self.direction
        reward = 0
        done = False

        # Check if the snake hits the wall or itself
        if not (0 <= new_head.x < number and 0 <= new_head.y < number) or new_head in self.snake_body[1:]:
            reward = -20
            done = True
            return self.get_state(), reward, done

        self.snake_body.insert(0, new_head)

        # Check if the snake eats the fruit
        if new_head == self.fruit:
            reward = 30
            self.score += 1
            self.place_fruit()
        else:
            self.snake_body.pop()

        return self.get_state(), reward, done

    def get_state(self):
        head = self.snake_body[0]
        left = head + LEFT
        right = head + RIGHT
        up = head + UP
        down = head + DOWN

        state = [
            # Danger straight
            (self.direction == UP and not self.is_safe(up)) or
            (self.direction == DOWN and not self.is_safe(down)) or
            (self.direction == LEFT and not self.is_safe(left)) or
            (self.direction == RIGHT and not self.is_safe(right)),

            # Danger right
            (self.direction == UP and not self.is_safe(right)) or
            (self.direction == DOWN and not self.is_safe(left)) or
            (self.direction == LEFT and not self.is_safe(up)) or
            (self.direction == RIGHT and not self.is_safe(down)),

            # Danger left
            (self.direction == UP and not self.is_safe(left)) or
            (self.direction == DOWN and not self.is_safe(right)) or
            (self.direction == LEFT and not self.is_safe(down)) or
            (self.direction == RIGHT and not self.is_safe(up)),

            # Move direction
            self.direction == LEFT,
            self.direction == RIGHT,
            self.direction == UP,
            self.direction == DOWN,

            # Fruit relative position
            self.fruit.x < head.x,  # Fruit left
            self.fruit.x > head.x,  # Fruit right
            self.fruit.y < head.y,  # Fruit up
            self.fruit.y > head.y,  # Fruit down
        ]

        return np.array(state, dtype=int)

    def is_safe(self, position):
        if position.x < 0 or position.x >= number or position.y < 0 or position.y >= number:
            return False  # Hit the wall
        if position in self.snake_body[1:]:
            return False  # Hit itself
        return True  # Safe

    def render(self):
        screen.fill((175, 215, 255))

        # Draw snake
        for block in self.snake_body:
            pygame.draw.rect(screen, BLUE, pygame.Rect(block.x * size, block.y * size, size, size))

        # Draw fruit
        pygame.draw.rect(screen, RED, pygame.Rect(self.fruit.x * size, self.fruit.y * size, size, size))

        # Draw score
        text = font.render(f'Score: {self.score}', True, BLACK)
        screen.blit(text, [10, 10])

        pygame.display.update()

class QLearningAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.q_table = np.zeros((2 ** state_size, action_size))
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.1
        self.gamma = 0.9  # Discount rate

    def get_state_index(self, state):
        return int("".join(map(str, state)), 2)

    def choose_action(self, state):
        state_index = self.get_state_index(state)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        return np.argmax(self.q_table[state_index])

    def update_q_value(self, state, action, reward, next_state):
        state_index = self.get_state_index(state)
        next_state_index = self.get_state_index(next_state)
        best_next_action = np.argmax(self.q_table[next_state_index])
        target = reward + self.gamma * self.q_table[next_state_index, best_next_action]
        self.q_table[state_index, action] += self.learning_rate * (target - self.q_table[state_index, action])

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def train_agent(episodes=8000):
    game = SnakeGame()
    agent = QLearningAgent(state_size=11, action_size=4)  # 11 inputs, 4 possible actions

    scores = []

    for e in range(episodes):
        state = game.reset()
        total_reward = 0

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            action = agent.choose_action(state)
            next_state, reward, done = game.step(action)
            agent.update_q_value(state, action, reward, next_state)

            state = next_state
            total_reward += reward

            if done:
                print(f"Episode: {e + 1}/{episodes}, Score: {game.score}, Epsilon: {agent.epsilon:.2f}")
                break

            if e % 100 == 0:
                game.render()
                clock.tick(FPS)

        scores.append(total_reward)
        agent.decay_epsilon()

    return scores, agent

if __name__ == "__main__":
    episodes = 8000
    scores, agent = train_agent(episodes)
    print(f"Training completed over {episodes} episodes.")
    print(f"Final Q-table:\n{agent.q_table}")
    print(f"Scores: {scores}")

    # Play the game with the trained agent
    game = SnakeGame()
    state = game.reset()
    agent.epsilon = 0  # Use the trained policy without exploration
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        action = agent.choose_action(state)
        next_state, reward, done = game.step(action)
        game.render()
        clock.tick(FPS)
        state = next_state
        if done:
            break

    pygame.quit()




