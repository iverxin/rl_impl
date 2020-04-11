import numpy as np

import ddqn2
import gym


env = gym.make('LunarLander-v2')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = ddqn2.DQNAgent(state_size, action_size)

done = False
batch_size = 32
agent.load("./checkpoint_ddqn/400")

for e in range(100):
    state = env.reset()
    state = np.reshape(state, [1, state_size])

    for time in range(1000):
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])
        state = next_state
        env.render()
        if(reward < -80):
            print('crashed!')
        if done:
            print(e)
            break
