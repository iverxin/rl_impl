# -*- coding: utf-8 -*-
from datetime import  datetime
import logging
import random
import gym
import numpy as np
from collections import deque

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import sendmail
import os
EPISODES = 1000

# gamma 0.95-0.99
# decay 0.996
# deque 2000- 100000
#batch size 32- 64
class DQNAgent:
    def __init__(self, state_size, action_size):
        # state size = 4
        self.state_size = state_size
        # action size = 2 left or right
        self.action_size = action_size
        # memory
        self.memory = deque(maxlen=1000000)
        self.gamma = 0.99  # discount rate
        self.epsilon = 1.0  # exploration rate

        self.epsilon_min = 0.01
        self.epsilon_decay = 0.996
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.batch_size = 64

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(150, input_dim=self.state_size, activation='relu'))
        # model.add(Dense(32,activation='relu'))
        model.add(Dense(120, activation='relu'))
        # out put layer, size = 2(left or right)
        model.add(Dense(self.action_size, activation='linear'))
        # create a model based on the information above
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # return a action
    def act(self, state):
        # epsilon-greed policy
        # if np.random.rand() <= self.epsilon:
        #     return random.randrange(self.action_size)

        # get a predict action from the NN network.
        act_values = self.model.predict(state)
        # argmax picks the index with the highest value. act_values[0] looks like [0.67, 0.2]
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        # sample some experiences from the memory, and train;
        start = datetime.now()
        minibatch = random.sample(self.memory, self.batch_size)

        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])

        states = np.squeeze(states,axis=1)
        next_states = np.squeeze(next_states)
        temp = np.amax(self.model.predict_on_batch(next_states),axis = 1)
        # 1-dones
        target = rewards + self.gamma * temp * (1-dones)

        target_formate = self.model.predict_on_batch(states)
        index = np.array([i for i in range(self.batch_size)])
        # Attention: the usage
        target_formate[index, actions] = target

        self.model.fit(states, target_formate, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    # env = gym.make('LunarLander-v2')
    env.seed(0)
    np.random.seed(0)
    logFileName = "dqn.log"
    logging.basicConfig(filename=logFileName, filemode='w', level=logging.INFO, format='%(asctime)s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)

    loss = deque(maxlen=100)
    # e = maxNum
    e = 0
    while e < EPISODES:
        e += 1
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        if e == int(EPISODES*0.3):
            sendmail.send('30% finished!, reward:'+str(total_reward))
        if e == int(EPISODES*0.6):
            sendmail.send('60% finished, reward:'+str(total_reward))
        if e == int(EPISODES*0.9):
            sendmail.send('90 finished, reward:'+str(total_reward))
        if e%10 == 0:
            agent.save('./checkpoint/'+str(e))
            print('check point{}'.format(e) )
        total_reward = 0
        for time in range(2000):
            env.render()
            # decide action
            # action: 0 none ,  1 right , 2 down, 3 left
            action = agent.act(state)
            # do the action and get reaction
            next_state, reward, done, _ = env.step(action)
            total_reward+=reward
            # reward
            # reward = reward if not done else -100
            next_state = np.reshape(next_state, [1, state_size])
            # memorize this anction and the real state return from env
            agent.memorize(state, action, reward, next_state, done)
            state = next_state
            if done:
                loss.append(total_reward)
                is_solved = np.mean(loss)
                print("episode: {}/{}, step: {}, e: {:.2}, coordinate:{:.3},{:.3}, reward:{:.3f}"
                      .format(e, EPISODES, time, agent.epsilon, state[0,0], state[0,1],total_reward))
                print("average:{}".format(is_solved))

                logging.info("episode: {}/{} - step: {}, e: {:.2} - coordinate:{:.3},{:.3} - reward:{:.5}"
                             .format(e, EPISODES, time, agent.epsilon, state[0, 0], state[0, 1], total_reward))
                break
            # train every 32(batch size);
            if len(agent.memory) > agent.batch_size:
                agent.replay(agent.batch_size)


    agent.save("./train_output/weights.h5")
    sendmail.send('Dear sir, the training mission has finished, please check it ^_^')