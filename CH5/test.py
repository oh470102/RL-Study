import gym

env = gym.make('CartPole-v1')
env.reset()

print(env.env.state)