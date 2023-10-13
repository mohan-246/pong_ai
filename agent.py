import pygame
import random
import numpy as np
from game import PongGame
import torch
from collections import deque
import os
import multiprocessing
from main import Linear_QNet , QTrainer
# Define your PongGame class here
MAX_MEMORY = 100_000
# Define the PongAgent class
class PongAgent:
    def __init__(self):
        self.epsilon = 160  # randomness
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()
        self.n_games = 0
        self.gamma = 0.5  # discount rate
        self.model = Linear_QNet(4 , 128 , 4)
        self.trainer = QTrainer(self.model, lr=0.001, gamma=self.gamma)
        
    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def get_state(self, game):
        # Define a method to get the game state similar to get_state in SnakeAgent
        state = [game.ball.x < game.player_paddle.x , game.ball.x > game.player_paddle.x , game.ball.y , game.player_paddle.x] 
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        # Define the remember method similar to SnakeAgent
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        # print(self.memory)
        if len(self.memory) > 1000:
            mini_sample = random.sample(self.memory, 1000) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        #for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)


    def train_short_memory(self, state, action, reward, next_state, done):
        # Define the train_short_memory method similar to SnakeAgent
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # Define the get_action method to choose actions based on the current state, similar to SnakeAgent
        self.epsilon = 160 - self.n_games
        final_move = [0, 0, 0 , 0]
        if random.randint(0, 160) < self.epsilon:
            # print("random")
            move = random.randint(0, 3)
            final_move[move] = 1
        else:
            # print("predicted")
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move
    # train_pong.py

def train():
    agent = PongAgent()
    record = 0
    n = 0
    while True:
        game = PongGame()
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old)
        done, score = game.update_game_state()
        reward = 0
        total_reward = reward

        while True:
            state_old = agent.get_state(game)
            done, score = game.update_game_state()
            
            if game.is_on_edge():
                final_move = agent.get_action(state_old)
                reward = game.play_step(final_move)
                state_new = agent.get_state(game)
                agent.train_short_memory(state_old, final_move, reward, state_new, done)
                agent.remember(state_old, final_move, reward, state_new, done)

            if done:
                total_reward += reward
                if score > record:
                    record = score
                    agent.model.save()
                game.reset()
                n += 1
                print(f'Score: {score} Reward: {reward} Game: {n} Record: {record}')  # Print the score and reward
                break

if __name__ == '__main__':
    train()
