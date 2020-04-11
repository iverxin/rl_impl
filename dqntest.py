# -*- coding: utf-8 -*-
import random
import gym
import numpy as np
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import sendmail
EPISODES = 1000


class DQNAgent:
    def __init__(self, state_size, action_size):
        # state size = 4
        self.state_size = state_size
        # action size = 2 left or right
        self.action_size = action_size
        # memory
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate

        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.01
        self.model = self._build_model()

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
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        # sample some experiences from the memory, and train;
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            # if done, make our target reward
            target = reward

            if not done:
                # predict the future discounted reward
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))

            # make the agent to approximately map
            # the current state to future discounted reward
            # We'll call that target_f，ndarray,size = 1,2
            # target_f[0,0] is the action 0 value; target_f[0,1]is action 1 value;

            target_f = self.model.predict(state)
            target_f[0][action] = target

            # Train the Neural Net with the state and target_f
            ret = self.model.fit(state, target_f, epochs=1, verbose=0)

        # make the epsilon(aka exploration rate) decay.
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return ret.history.get('loss')
    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":
    # env = gym.make('CartPole-v1')
    env = gym.make('LunarLander-v2')
    env.seed()
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    agent.load("./checkpoint_ddqn/1000")
    done = False
    batch_size = 64
    loss = 0.1
    for e in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        total_reward = 0;
        for time in range(10000):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            env.render()
            total_reward+=reward

            if reward< -30 :
                print(reward)
            next_state = np.reshape(next_state, [1, state_size])
            state = next_state
            if done:
                print("episode: {}/{}, step: {}, e: {:.2}, coordinate:{:.3},{:.3}, reward:{}"
                      .format(e, EPISODES, time, agent.epsilon, state[0,0], state[0,1],total_reward))
                print("loss:{}".format(loss))
                break