import shutil
import sys
import time
from collections import deque
import datetime
import random
import tensorlayer as tl
import argparse
import gym
import tensorflow as tf
from tensorflow import keras

import numpy as np
import os
import matplotlib.pyplot as plt
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
print(tf.__version__)
BASE_LOG_DIR = './ddpg_log'
ALG_NAME = 'DDPG'
ENV = 'LunarLanderContinuous-v2'
# ENV = 'MountainCarContinuous-v0'
RANDOM_SEED = 2

EPISODES = 1000
MAX_STEPS = 200
LR_ACTOR = 0.001
LR_CRITIC = 0.002
W_init = tf.random_normal_initializer(mean=0, stddev=0.3)
b_init = tf.constant_initializer(0.1)

class aDDPG(object):
    def __init__(self, state_dim, action_dim, action_bound):
        self.min_exploration = 0.05
        self.GAMMA = 0.99
        self.exploration = 2
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.actor_lr = 0.001
        self.critic_lr = 0.002
        self.tau = 0.001
        self.batch_size = 32
        self.memory = deque(maxlen=10000)
        self.W_INIT = tf.random_normal_initializer(mean=0, stddev=0.3)
        self.B_INIT = tf.constant_initializer(0.1)

        input_state_shape = state_dim
        input_action_shape = action_dim

        # Actor Network
        self.actor = self.create_actor_network(input_state_shape)
        # Critic Net
        self.critic = self.create_critic_network(input_state_shape, input_action_shape)

        # Actor target Network
        self.actor_target = self.create_actor_network(input_state_shape, 'target')
        self.copy_parameters(self.actor, self.actor_target)

        # Critic target Network
        self.critic_target = self.create_critic_network(input_state_shape, input_action_shape, "target")
        self.copy_parameters(self.critic, self.critic_target)


        self.ema = tf.train.ExponentialMovingAverage(decay=1 - self.tau)
        self.actor_opt = tf.optimizers.Adam(self.actor_lr)
        self.critic_opt = tf.optimizers.Adam(self.critic_lr)

        # Periodically updating target network with online network


    def create_actor_network(self, input_state_shape, name=""):
        model = keras.Sequential(name='Actor ' + name)
        model.add(keras.layers.Dense(units=200, input_dim=input_state_shape, activation=tf.nn.relu,
                                     kernel_initializer=self.W_INIT, bias_initializer=self.B_INIT))
        model.add(keras.layers.Dense(units=200, activation=tf.nn.relu, kernel_initializer=self.W_INIT,
                                     bias_initializer=self.B_INIT))
        model.add(keras.layers.Dense(units=self.action_dim, activation=tf.nn.tanh, name='output',
                                     kernel_initializer=self.W_INIT, bias_initializer=self.B_INIT))
        model.add(keras.layers.Lambda(lambda x: self.action_bound * x))
        print(model.summary())

        return model

    def create_critic_network(self, input_state_shape, input_action_shape, name=''):

        status_input = keras.layers.Input(shape=input_state_shape, name='critic_state_input')
        actions_input = keras.layers.Input(shape=input_action_shape, name='critic_action_input')
        # concat = keras.layers.Concatenate(axis=1)([status_input, actions_input])
        concat = tf.concat(([status_input, actions_input]),axis=1, name = "concat")

        layer = keras.layers.Dense(units=200, activation=tf.nn.relu,
                                   kernel_initializer=self.W_INIT, bias_initializer=self.B_INIT)(concat)

        layer = keras.layers.Dense(units=200, activation=tf.nn.relu, kernel_initializer=self.W_INIT,
                                   bias_initializer=self.B_INIT)(layer)

        out = keras.layers.Dense(units=1, kernel_initializer=self.W_INIT,
                                     bias_initializer=self.B_INIT,name='output_Q')(layer)

        model = keras.models.Model(inputs=[status_input, actions_input], outputs=out, name='Critic ' + name)

        print(model.summary())

        return model

    def copy_parameters(self, src_model, tar_model):
        zipped = zip(src_model.trainable_weights, tar_model.trainable_weights)
        for i, j in zipped:
            # tf.assign
            j.assign(i)

    def act(self, state, greedy=False):

        a = self.actor(np.array([state], dtype=np.float32))[0]
        for i in tf.math.is_nan(a):
            if i:
                print("action is nan")
                os._exit(0)
        if greedy:
            return a
        # add noise
        a = np.random.normal(a, self.exploration)
        # assert the a is in action bound
        a = np.clip(a, -self.action_bound, self.action_bound)
        return a


    # def eget_action(self, s, greedy=False):
    #     """
    #     Choose action
    #     :param s: state
    #     :param greedy: get action greedy or not
    #     :return: act
    #     """
    #     a = self.actor(np.array([s], dtype=np.float32))[0]
    #     if greedy:
    #         return a
    #     return np.clip(
    #         np.random.normal(a, self.exploration), -self.action_bound, self.exploration
    #     )  # add randomness to action selection for exploration

    def store(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        """
        Replay, learn from the experience.
        :return: None
        """
        if self.exploration > self.min_exploration:
            self.exploration *= 0.9995
        minibatch = random.sample(self.memory, self.batch_size)

        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        actions = actions.astype(dtype='float32')
        rewards = np.array([i[2] for i in minibatch])
        rewards = np.reshape(rewards, (self.batch_size, 1))
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])

        # train critic network
        with tf.GradientTape() as tape:
            next_actions = self.actor_target(next_states)
            q_ = self.critic_target([next_states, next_actions])
            y = rewards + self.GAMMA * q_  # y target.
            q = self.critic([states, actions])  # q predicted
            loss = tf.losses.mean_squared_error(y, q)
        critic_grads = tape.gradient(loss, self.critic.trainable_weights)
        self.critic_opt.apply_gradients(zip(critic_grads, self.critic.trainable_weights))

        # train actor network
        with tf.GradientTape() as tape:
            a = self.actor(states)
            q = self.critic([states, a])  # q is a function of a
            act_j = -tf.reduce_mean(q)  # max the q
        actor_grads = tape.gradient(act_j, self.actor.trainable_weights)
        self.actor_opt.apply_gradients(zip(actor_grads, self.actor.trainable_weights))

        # tensorboard
        critic_loss_mean(loss)
        actor_j_mean(act_j)

        # update to target network
        # self.ema_update()
        self.update_para()
    def ema_update(self):
        """
        Soft updating by exponential smoothing
        :return: None
        """
        # union two list int paras
        paras = self.actor.trainable_weights + self.critic.trainable_weights
        self.ema.apply(paras)
        target = self.actor_target.trainable_weights + self.critic_target.trainable_weights
        for i, j in zip(target, paras):
            i.assign(self.ema.average(j))

    def update_para(self):
        src_paras = self.actor.trainable_weights + self.critic.trainable_weights
        tar_paras = self.actor_target.trainable_weights + self.critic.trainable_weights
        for i, j in zip(src_paras, tar_paras):
            j.assign(self.tau*i+(1-self.tau)*j)

    def save(self):
        path = os.path.join('model', '_'.join([ALG_NAME, ENV]))
        if not os.path.exists(path):
            os.makedirs(path)
        tl.files.save_weights_to_hdf5(os.path.join(path, 'actor_net.hdf5'), self.actor)
        tl.files.save_weights_to_hdf5(os.path.join(path, 'actor_target.hd5'), self.actor_target)
        tl.files.save_weights_to_hdf5(os.path.join(path, 'critic.hd5'), self.actor_target)
        tl.files.save_weights_to_hdf5(os.path.join(path, 'critic_target.hd5'), self.actor_target)
        print("Save weights")

    def load(self):
        path = os.path.join('model', '_'.join([ALG_NAME, ENV]))
        if not os.path.exists(path):
            os.makedirs(path)
        try:
            tl.files.load_hdf5_to_weights_in_order(os.path.join(path, 'actor_net.hdf5'), self.actor)
            tl.files.load_hdf5_to_weights_in_order(os.path.join(path, 'actor_target.hd5'), self.actor_target)
            tl.files.load_hdf5_to_weights_in_order(os.path.join(path, 'critic.hd5'), self.actor_target)
            tl.files.load_hdf5_to_weights_in_order(os.path.join(path, 'critic_target.hd5'), self.actor_target)
            print("Load weights")
        except Exception as e:
            print(e)




if __name__ == "__main__":

    # parameter process
    args = sys.argv
    if len(args) > 1 and args[1] == 'reset':
        if os.path.exists(BASE_LOG_DIR):
            shutil.rmtree(BASE_LOG_DIR)
            print('Delete the log')


    env = gym.make(ENV)

    # random seed
    env.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    """
    agent:
        action:
            first: main engin : -1-0 off,  0-1: on 
            second: -1 to -0.5 right
                    -0.5 to 0.5 off
                    0.5 to 1 left

        statue:

    """

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high

    agent = aDDPG(state_dim, action_dim, action_bound)

    t0 = time.time()

    # ===========tensorboard=================
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = os.path.join(BASE_LOG_DIR, 'train')

    # define Total_rewards
    total_rewards_mean = keras.metrics.Mean('total_rewards_mean', dtype=tf.float32)
    critic_loss_mean = keras.metrics.Mean('critic_loss_mean', dtype=tf.float32)
    actor_j_mean = keras.metrics.Mean('actor_j_mean', dtype=tf.float32)

    train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    episode = agent.load()

    if episode == None:
        episode = 0

    for episode in range(EPISODES):
        if (episode) % 10 == 0:
            pass
            # agent.save(episode)
            # if agent.exploration > agent.min_exploration:
            #     agent.exploration = agent.exploration * 0.9995
        if episode %30 ==0 :
            # agent.ema_update()
            agent.update_para()

        state = env.reset()
        episode_reward = 0
        for t in range(MAX_STEPS):
            env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.store(state, action, reward, next_state, done)

            state = next_state

            episode_reward += reward

            if len(agent.memory) > agent.batch_size:
                agent.replay()
            if done:
                break

        print('Training  | Episode: {}/{}  | Episode Reward: {:.4f}  | exporation: {:.4f}'.format(
            episode, EPISODES, episode_reward,
            agent.exploration)
        )
        # tensorboard
        total_rewards_mean(episode_reward)

        # graph name
        with train_summary_writer.as_default():
            tf.summary.scalar('reward', total_rewards_mean.result(), step=episode)

