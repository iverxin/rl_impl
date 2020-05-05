# -*- coding: utf-8 -*-
import os
import random
import gym
import numpy as np
from collections import deque
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import logging

import sendmail
"""
Author: Spade
@Time : 2020/5/5 
@Email: spadeaiverxin@163.com
"""
EPISODES = 1000

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=1000000)
        self.gamma = 0.99    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        self.batch_size = 64

    """Huber loss for Q Learning

    References: https://en.wikipedia.org/wiki/Huber_loss
                https://www.tensorflow.org/api_docs/python/tf/keras/losses/Huber
    """

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(1000, input_dim=self.state_size, activation='relu'))
        model.add(Dense(1000, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])
        states = np.squeeze(states)
        next_states = np.squeeze(next_states)

        # a: batch size x 4
        actions_temp = self.model.predict_on_batch(next_states)
        actions_temp = np.argmax(actions_temp,axis=1) # bach size

        target_f = self.model.predict_on_batch(states) # batch size x 4

        # t: batch size x 4
        t = self.target_model.predict_on_batch(next_states)
        ind = [i for i in range(batch_size)]
        t = t[ind,actions_temp]
        target = rewards + self.gamma * t * (1-dones)
        target_f[ind, actions] = target

        self.model.fit(states,target_f,epochs=1,verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

    def info(self):
        return "INFO: DDQN, Random off, seed(0)" + \
            "batch size:" + str(self.batch_size)+ \
            "learning rate:" + str(self.learning_rate)


if __name__ == "__main__":

    logFileName = './logging.txt'
    env = gym.make('LunarLander-v2')
    env.seed(0)
    np.random.seed(0)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    done = False

    logging.basicConfig(filename=logFileName, filemode='w', level=logging.INFO, format='%(asctime)s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S')

    print(agent.info())
    logging.info(agent.info())

    checkpoint_dir = './checkpoint_ddqn/'
    try:
        fileList = os.listdir(checkpoint_dir)
        maxNum = 0
        for f in fileList:
            if maxNum < int(f):
                maxNum = int(f)
        agent.load(checkpoint_dir + str(maxNum))
        agent.epsilon = 0.01
        print("load the chekpoint of {}".format(maxNum))
    except  Exception as err:
        print('no backup')
    done = False

    loss = 0.1
    e = maxNum
    while e < EPISODES:
        e += 1
        state = env.reset()

        state = np.reshape(state, [1, state_size])
        if e == int(EPISODES * 0.3):
            sendmail.send('30% finished!, reward:' + str(total_reward))
        if e == int(EPISODES * 0.6):
            sendmail.send('60% finished, reward:' + str(total_reward))
        if e == int(EPISODES * 0.9):
            sendmail.send('90 finished, reward:' + str(total_reward))
        if e % 10 == 0:
            agent.save(checkpoint_dir + str(e))
            print('check point{}'.format(e))
        total_reward = 0
        for time in range(1000):
    #        env.render()
            # decide action
            # action: 0 none ,  1 right , 2 down, 3 left
            action = agent.act(state)
            # do the action and get reaction
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            # reward
            # reward = reward if not done else -100
            next_state = np.reshape(next_state, [1, state_size])
            # memorize this anction and the real state return from env
            agent.memorize(state, action, reward, next_state, done)
            state = next_state

            if done:
                print("episode: {}/{}, step: {} - e: {:.2} - coordinate:{:.3},{:.3} - reward:{:.5}"
                      .format(e, EPISODES, time, agent.epsilon, state[0, 0], state[0, 1], total_reward))
                logging.info("episode: {}/{} - step: {}, e: {:.2} - coordinate:{:.3},{:.3} - reward:{:.5}"
                      .format(e, EPISODES, time, agent.epsilon, state[0, 0], state[0, 1], total_reward))
                break

            if len(agent.memory) > agent.batch_size:
                loss = agent.replay(agent.batch_size)
        agent.update_target_model()
    agent.save("./train_output/weights.h5")
    sendmail.send('Dear sir, the'
                  ' training mission has finished, please check it ^_^')