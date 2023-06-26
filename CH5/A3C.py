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
epochs, n_workers = 500, 4
params = {'epochs': epochs, 'n_workers': n_workers}
counter = mp.Value('i', 0)  # 'i' indicates integer type

def train_model():
    try:
        for i in range(params['n_workers']):
            p = mp.Process(target=worker, args=(i, MasterNode, counter, params, queue))
            p.start()
            processes.append(p)

        print("Joining processes...")
        for p in processes:
            p.join()
        print("Joining complete.")

    except Exception as e:
        print("Exception occurred:", str(e))
        # Handle or log the exception appropriately

    finally:
        print("Terminating processes...")
        for p in processes:
            p.terminate()
        print("All processes have terminated.")

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
    train_model()
    print("training over")

    scores = []
    while not queue.empty():
        scores.append(queue.get())

    plt.plot(moving_average(scores, 50, reg=False))
    plt.show()

    #test_model()
    






