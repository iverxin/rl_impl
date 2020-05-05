import argparse
import os
import sys
import threading
import tensorflow.keras as keras
import datetime
import gym
import numpy as np
import tensorflow as tf
import tensorflow_probability as tp

from ac import model

"""
Global var
"""
ENV = 'Pendulum-v0'
TRAJ_LEN = 10
GAMMA = 0.9
LR_A = 0.0001
LR_C = 0.001
ENTROY_BETA = 0.01
GLOBAL_STEPS = 0
MAX_GLOBAL_EP = 3000
SEED = 2
EPISODE_STEP = 200


class A3C(object):
    def __init__(self, name):
        global train_summary_writer
        self.state_dim = STATE_DIM
        self.action_dim = ACTION_DIM
        self.action_bound = ACTION_BOUND
        self.name = name

        # create net work
        self.policy_net = model.create_actor_network(self.state_dim, self.action_dim, self.action_bound,
                                                     scope=self.name)
        self.value_net = model.create_critic_network(input_state_shape=self.state_dim,
                                                     input_action_shape=self.action_dim, scope=self.name)
        # with train_summary_writer.as_default():
        #     tf.summary.trace_export(name='workers graph',step=0, profiler_outdir=train_log_dir)
        self.memory = []

    def get_action(self, _state, greedy=False):
        """
        Get an action
        :param greedy:
        :param _state:
        :return:
        """
        # get mu, sigma
        mu, sigma = self.policy_net(np.array([_state]))

        # scale
        mu, sigma = mu * self.action_bound, sigma + 1e-4

        # if use greedy policy, action is mu
        if greedy:
            return np.squeeze(mu, 0)
        if sigma < 0:
            print('sigma<0')
            sys.exit(0)
        # create normal distribution and explore

        norm_dist = tp.distributions.Normal(mu, sigma)

        # sample actions from distributions, with exploration
        actions = norm_dist.sample(seed=SEED)
        actions = tf.clip_by_value(actions, -self.action_bound, self.action_bound)
        return np.squeeze(actions, axis=0)

    def value(self, state):
        state = np.asarray([state], dtype=np.float32)
        return self.value_net(state)[0]

    @tf.function
    def update_to_global(self, states, actions, rets):
        """
        :param states: a batch state
        :param actions: a batch action
        :param rets: a batch return
        :return:
        """
        global GLOBAL_AC
        global_AC_w_lock.acquire()

        with tf.GradientTape() as tape:
            td = tf.subtract(rets, self.value_net(states), name='td-err')
            loss = tf.reduce_mean(tf.square(td), name='critic_loss')
            # loss = tf.losses.mean_squared_error(rets, self.value_net(states))
        v_grad = tape.gradient(loss, self.value_net.trainable_weights)
        OPT_C.apply_gradients(zip(v_grad, GLOBAL_AC.value_net.trainable_weights))

        with tf.GradientTape() as tape:
            adv = tf.subtract(rets, self.value_net(states))

            # compute action's proability
            mu, sigma = self.policy_net(states)
            mu, sigma = self.action_bound * mu, sigma + 1e-4  # 防止sigma = 0,导致nan
            norm_dist = tp.distributions.Normal(mu, sigma)
            log_prob = norm_dist.log_prob(actions)
            expect_v = log_prob * adv

            # add entropy to encourage exploration
            expect_v = ENTROY_BETA * norm_dist.entropy() + expect_v

            # gradient ascent
            p_loss = tf.reduce_mean(-expect_v)  # this is a batch, no only one.
        p_grad = tape.gradient(p_loss, self.policy_net.trainable_weights)
        OPT_A.apply_gradients(zip(p_grad, GLOBAL_AC.policy_net.trainable_weights))
        # release the GLOBAL_AC
        global_AC_w_lock.release()

    @tf.function
    def pull_from_global(self):
        global_AC_w_lock.acquire()
        for local_para, global_para in zip(self.value_net.trainable_weights, GLOBAL_AC.value_net.trainable_weights):
            local_para.assign(global_para)

        for local_para, global_para in zip(self.policy_net.trainable_weights, GLOBAL_AC.policy_net.trainable_weights):
            local_para.assign(global_para)
        global_AC_w_lock.release()

    def save(self, name=''):
        global_AC_w_lock.acquire()
        path = os.path.join('a3c', ENV)
        if not os.path.exists(path):
            os.makedirs(path)

        self.policy_net.save_weights(os.path.join(path, 'policy_net' + name))
        self.value_net.save_weights(os.path.join(path, 'value_net' + name))
        global_AC_w_lock.release()

    def load(self):
        path = os.path.join('a3c', ENV)
        self.policy_net.load_weights(os.path.join(path, 'policy_net'))
        self.value_net.load_weights(os.path.join(path, 'value_net'))


class Worker(object):
    def __init__(self, name):
        self.env = gym.make(ENV)
        # self.env.seed(SEED)
        self.name = name
        self.agent = A3C(name)
        self.step_counter = 0
        self.total_reward = 0

    def get_trajectory(self, init_state):
        """

        :param init_state:
        :return: trajectory[states, actions, rewards], terminal(boolean)
        """
        # memory
        m_states = []
        m_actions = []
        m_reward = []
        global GLOBAL_STEPS
        # get MINI_STEP trajectory
        for i_ in range(TRAJ_LEN):
            self.step_counter += 1
            m_states.append(init_state)

            action = self.agent.get_action(init_state)
            # if self.name == 'worker1':
            #     env.render()
            next_state, reward, done, _ = self.env.step(action)
            reward = reward
            self.total_reward += reward
            m_actions.append(action)
            m_reward.append(reward)

            init_state = next_state

            if done:
                GLOBAL_STEPS += 1
                return [m_states, m_actions, m_reward, None], True
        return [m_states, m_actions, m_reward, init_state], False

    def learn(self):
        # get trajectory
        global GLOBAL_STEPS
        # global EPISODE_STEP
        state = self.env.reset()
        while not COORD.should_stop():
            if GLOBAL_STEPS > MAX_GLOBAL_EP:
                print(self.name + " out!")
                break
            if (GLOBAL_STEPS+1) % 500 == 0:
                GLOBAL_AC.save(name=str(GLOBAL_STEPS))
                print("save the ckpoint")

            traj, terminal = self.get_trajectory(state)

            states, actions, rewards, next_state = traj[0], traj[1], traj[2], traj[3]
            if terminal:
                v = 0.
                print(self.name + " total reward:{}, episode{}".format(self.total_reward, GLOBAL_STEPS))
                # tensorboard
                with train_summary_writer.as_default():
                    tf.summary.scalar('reward', self.total_reward, step=GLOBAL_STEPS)
                self.total_reward = 0
                state = self.env.reset()
            else:
                v = self.agent.value_net(tf.constant([next_state]))
                v = tf.squeeze(v, 0)
                state = next_state
            # compulate return list

            num = len(states)
            ret = v
            ret_list = []
            for i_ in range(num).__reversed__():
                ret = rewards[i_] + GAMMA * ret
                ret_list.append(ret)
            ret_list.reverse()

            states = tf.convert_to_tensor(states, dtype=tf.float32)
            actions = tf.convert_to_tensor(actions, dtype=tf.float32)
            ret_list = tf.convert_to_tensor(ret_list, dtype=tf.float32)

            self.agent.update_to_global(states, actions, ret_list)
            self.agent.pull_from_global()


if __name__ == "__main__":
    # add arguments in command  --train/test
    parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
    parser.add_argument('--train', dest='train', action='store_true', default=True)
    parser.add_argument('--test', dest='test', action='store_true', default=True)
    args = parser.parse_args()

    env = gym.make(ENV)
    env.seed(SEED)
    STATE_DIM = env.observation_space.shape[0]
    ACTION_DIM = env.action_space.shape[0]
    ACTION_BOUND = env.action_space.high
    # NUM_WORKERS = multiprocessing.cpu_count()
    NUM_WORKERS = 8
    print('num workers:{}'.format(NUM_WORKERS))
    # global net
    total_reward = 0

    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    GLOBAL_AC = A3C('Global name')

    if not os.path.exists('a2c'):
        os.makedirs('a2c')
    BASE_LOG_DIR = 'a2c'

    # lock
    global_AC_w_lock = threading.Lock()

    # train
    if args.train:
        ##########tensorboard##############
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + 'a3c'
        train_log_dir = os.path.join(BASE_LOG_DIR, current_time)
        # define tensorboard Mean
        critic_loss_mean = keras.metrics.Mean('critic_loss_mean', dtype=tf.float32)
        actor_j_mean = keras.metrics.Mean('actor_j_mean', dtype=tf.float32)
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)

        with tf.device("/cpu:0"):
            OPT_A = tf.optimizers.RMSprop(LR_A, name='RMSPropA')
            OPT_C = tf.optimizers.RMSprop(LR_C, name='RMSPropC')
            workers = []

            # Create worker
            for i in range(NUM_WORKERS):
                i_name = 'worker{}'.format(i)
                workers.append(Worker(name=i_name))

            # 协调员
            COORD = tf.train.Coordinator()

            workers_threads = []
            for worker in workers:
                job = lambda: worker.learn()
                t = threading.Thread(target=job)
                t.start()
                workers_threads.append(t)
            COORD.join(workers_threads)
            GLOBAL_AC.save()

    if args.test:
        print('loading weights')
        GLOBAL_AC.load()

        for ep in range(10):
            s = env.reset()
            ep_reward = 0
            while True:
                env.render()
                a = GLOBAL_AC.get_action(s, greedy=True)
                s, r, d, _ = env.step(a)
                r = r
                ep_reward += r

                if d:
                    break

            print(
                "Testing: Episode:{}/{}, reward {:.4f}".format(ep + 1, 10, ep_reward)
            )
