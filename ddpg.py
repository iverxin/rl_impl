import datetime
import os
import random
from collections import deque
import time

import gym
import tensorlayer as tl
import tensorflow as tf
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
ALG_NAME='DDPG'
ENV = 'LunarLanderContinuous-v2'
RANDOM_SEED = 2
EPISODES = 1000
MAX_STEPS = 200
W_init = tf.random_normal_initializer(mean=0, stddev=0.3)
b_init = tf.constant_initializer(0.1)


LR_A = 0.001  # learning rate for actor
LR_C = 0.002  # learning rate for critic # 0.002
GAMMA = 0.99  # reward discount #0.99
TAU = 0.01  # soft replacement
MEMORY_CAPACITY = 10000  # size of replay buffer
BATCH_SIZE = 32  # update action batch size
VAR = 2  # control exploration


class DDPG(object):

    def __init__(self, state_dim, action_dim, action_bound, learning_rate=0.001, batch_size=64):
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

        input_state_shape = [None, state_dim]
        input_action_shape = [None, action_dim]

        # Actor Network
        self.actor = self.get_actor(input_state_shape)
        self.actor.train()

        # Critic Net
        self.critic = self.get_critic(input_state_shape, input_action_shape)
        self.critic.train()

        # Actor target Network
        self.actor_target = self.get_actor(input_state_shape, 'target')
        self.copy_parameters(self.actor, self.actor_target)
        self.actor_target.eval()

        # Critic target Network
        self.critic_target = self.get_critic(input_state_shape, input_action_shape, "target")
        self.copy_parameters(self.critic, self.critic_target)
        self.critic_target.eval()

        self.ema = tf.train.ExponentialMovingAverage(decay=1 - self.tau)
        self.actor_opt = tf.optimizers.Adam(self.actor_lr)
        self.critic_opt = tf.optimizers.Adam(self.critic_lr)

        # Periodically updating target network with online network

    def get_actor(self, input_state_shape, name=""):
        ni = tl.layers.Input(shape=input_state_shape, name="input")

        nn = tl.layers.Dense(n_units=200, act=tf.nn.relu, name='dense1',
                             W_init=W_init, b_init = b_init)(ni)

        nn = tl.layers.Dense(n_units=200, act=tf.nn.relu, name="dense2",
                             W_init=W_init, b_init = b_init)(nn)

        nn = tl.layers.Dense(n_units=self.action_dim, act=tf.nn.tanh, name="output",
                             W_init=W_init, b_init = b_init)(nn)

        nn = tl.layers.Lambda(lambda x: self.action_bound * x)(nn)

        network = tl.models.Model(inputs=ni, outputs=nn, name="Actor" + name)
        print(network)

        return network

    def get_critic(self, input_state_shape, input_action_shape, name=''):
        state_input = tl.layers.Input(input_state_shape, name='critic state input')
        action_input = tl.layers.Input(input_action_shape, name='critic action input')
        nn = tl.layers.Concat(concat_dim=1)([state_input, action_input])
        nn = tl.layers.Dense(n_units=200, act=tf.nn.relu, W_init=W_init, b_init = b_init)(nn)
        nn = tl.layers.Dense(n_units=200, act=tf.nn.relu, W_init=W_init, b_init = b_init)(nn)
        nn = tl.layers.Dense(n_units=1, W_init=W_init, b_init = b_init, name='output,Q')(nn)
        network = tl.models.Model(inputs=[state_input, action_input], outputs=nn, name='Critic' + name)
        print(network)
        return network

    def copy_parameters(self, src_model, tar_model):
        zipped = zip(src_model.trainable_weights, tar_model.trainable_weights)
        for i, j in zipped:
            # tf.assign
            j.assign(i)

    def get_action(self, state, greedy=False):

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


    def eget_action(self, s, greedy=False):
        """
        Choose action
        :param s: state
        :param greedy: get action greedy or not
        :return: act
        """
        a = self.actor(np.array([s], dtype=np.float32))[0]
        if greedy:
            return a
        return np.clip(
            np.random.normal(a, self.exploration), -self.action_bound, self.exploration
        )  # add randomness to action selection for exploration

    def store(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))


    ## Safe
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
            td_error = tf.losses.mean_squared_error(y, q)
        critic_grads = tape.gradient(td_error, self.critic.trainable_weights)
        self.critic_opt.apply_gradients(zip(critic_grads, self.critic.trainable_weights))

        # train actor network
        with tf.GradientTape() as tape:
            a = self.actor(states)
            q = self.critic([states, a])  # q is a function of a
            act_j = -tf.reduce_mean(q)  # max the q
        actor_grads = tape.gradient(act_j, self.actor.trainable_weights)
        self.actor_opt.apply_gradients(zip(actor_grads, self.actor.trainable_weights))

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

    env = gym.make(ENV)

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

    agent = DDPG(state_dim=state_dim, action_dim = action_dim, action_bound = action_bound)
    # agent.load()
    t0 = time.time()
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = './tb_log/gradient_tape/' + 'train'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    tl.utils.open_tensorboard('./tb_log/gradient_tape/', port=6006)

    for episode in range(EPISODES):
        if episode+1 % 10 == 0:
            # agent.save()
            pass
        state = env.reset()
        episode_reward = 0
        # print(episode_reward)
        for t in range(MAX_STEPS):
            env.render()
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.store(state, action, reward, next_state, done)
            episode_reward += reward
            state = next_state

            if len(agent.memory) > 32:
                agent.replay()
            if done:
                break
        print('Training  | Episode: {}/{}  | Episode Reward: {:.4f}  | exporation: {:.4f}'.format(
            episode + 1, EPISODES, episode_reward,
            agent.exploration)
        )
