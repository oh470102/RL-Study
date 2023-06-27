### A3C 는 MC로 사용했을 땐 분산이 크고 학습 과정에 많은 fluctuation이 있다
### 이를 N-step 기법을 통해 완화하는 편이며 실제로 큰 효과가 있는 듯 하다.

### NOTE: Deadlock happens when calling p.join() before emptying queue.
### queue에서 item을 다 retrieve하고 join()해야 정상적으로 작동한다.

from Model import ActorCritic
from A3C_utils import *
import torch
import numpy as np
import gym
import torch.multiprocessing as mp
import matplotlib.pyplot as plt

### MATPLOTLIB 오류 해결
matplotlib_error()

### TRAINING CARTPOLE WITH A3C
MasterNode = ActorCritic()
MasterNode.share_memory()
processes = []
queue = mp.Queue()
epochs, n_workers = 1000, 5
params = {'epochs': epochs, 'n_workers': n_workers}
counter = mp.Value('i', 0)  # 'i' indicates integer type

def train_model():

    global processes
    scores = []

    for i in range(params['n_workers']):
        p = mp.Process(target=worker, args=(i, MasterNode, counter, params, queue))
        p.start()
        processes.append(p)

    count = 0
    while count < n_workers * epochs:
        count += 1
        scores.append(queue.get())

    for p in processes:
        p.join()

    return scores

def test_model():
    env = gym.make('CartPole-v1', render_mode='human')
    env.reset()
    env.render()
    curr_state = torch.from_numpy(env.env.state).float()
    done = False

    while not done:
        logits, values = MasterNode(curr_state)
        action = torch.distributions.Categorical(logits=logits).sample()
        next_state, reward, done, _, _ = env.step(action.detach().numpy())
        if done: break

        curr_state = torch.from_numpy(next_state).float()

if __name__ == '__main__':
    scores = train_model()

    plt.plot(moving_average(scores, 10, reg=False))
    plt.show()

    #test_model()
    






