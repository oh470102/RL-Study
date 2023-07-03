### Implementation of distributional Q-learning to solve atari freeway
### The algorithm from the book seems to be slightly different
### from the original one

from DistQlearning_utils import *
from collections import deque
from random import shuffle
import matplotlib.pyplot as plt
import torch
import numpy as np
import gym

### ENV
env = gym.make('Freeway-ram-v4')
aspace = 3

### HYPERPARAMS
vmin, vmax = -10, 10
replay_size = 1024
batch_size = 32
nsup = 51
dz = (vmax-vmin)/(nsup-1)
support = torch.linspace(vmin, vmax, nsup)

replay = deque(maxlen=replay_size)
lr = 5e-4
gamma = 0.1
epochs = 10
eps = 0.3           # starting epsilon for e-greedy
eps_min = 0.05      # min. epsilon to decay
priority_level = 10 # for PER
update_freq = 25    # target network update frequency

### MODEL
tot_params = 128 * 100 + 25 * 100 + aspace * 25 * 51
theta = torch.randn(tot_params)/10.  # initialize random params.
theta.requires_grad = True
theta_2 = theta.detach().clone()      # target_network

losses = []
renders = []
state = preproc_state(env.reset()[0])

### TRAINING
for i in range(epochs):
    env = gym.make('Freeway-ram-v4')
    state, _ = env.reset()
    done = False
    j = 0
    print(f"currently on epoch {i}/{epochs}")

    while not done:
        j += 1
        state = preproc_state(state)
        pred = dist_dqn(state, theta, aspace=aspace)

        if i < replay_size or np.random.rand(1) < eps:
            action = env.action_space.sample()
        else:
            action = get_action(pred.unsqueeze(dim=0).detach(), support).item()

        state2, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        # too many steps
        if j > 250: 
            done = True
            reward = -10

        # Scaling of rewards
        reward = 10 if reward == 1 else reward
        reward = -10 if done else reward
        reward = - 1 if reward == 0 else reward 

        exp = (state, action, reward, torch.from_numpy(state2))
        replay.append(exp)

        # simple Prioritized Experience Replay: -> just put more copies of the good exps.
        if reward == 10:
            for e in range(priority_level):
                replay.append(exp)

        shuffle(replay)
        state = state2

        if len(replay) == replay_size:
            indx = np.random.randint(low=0, high=len(replay), size=batch_size) # choose exps
            exps = [replay[j] for j in indx]

            state_batch = torch.stack([ex[0] for ex in exps], dim=1).squeeze()
            action_batch = torch.Tensor([ex[1] for ex in exps])
            reward_batch = torch.Tensor([ex[2] for ex in exps])
            state2_batch = torch.stack([ex[3] for ex in exps], dim=1).squeeze()
            
            # calculates target, don't forget to detach!
            pred_batch = dist_dqn(state_batch.detach(), theta, aspace=aspace)
            pred2_batch = dist_dqn(state2_batch.detach().view(batch_size, -1).float(), theta_2, aspace)
            target_dist = get_target_dist(pred2_batch, action_batch, reward_batch,
                                        support, lim=(vmin, vmax), gamma=gamma)
            
            # calculate loss, backpropagate
            loss = lossfn(pred_batch, target_dist.detach())
            losses.append(loss.item())
            loss.backward()
            with torch.no_grad():
                theta -= lr * theta.grad
            theta.requires_grad = True    # 와! optimizer 굳이 안 쓰기!

        
    if i % update_freq == 0:
        theta_2 = theta.detach().clone()

    if i > 100 and eps > eps_min:
        dec = 1./np.log2(i)
        dec = dec / 1e3
        eps -= dec

# Show loss graph
resolve_matplotlib_error()
plt.plot(losses)
plt.show()

# Show a test case
for i in range(3):
    done = False
    env = gym.make('Freeway-ram-v4', render_mode='human')
    state, _ = env.reset()

    while not done:
        state = preproc_state(state)
        pred = dist_dqn(state, theta, aspace=aspace)
        action = get_action(pred.unsqueeze(dim=0).detach(), support).item()
        state, reward, terminated, truncated, info = env.step(action)
        print(info)

        done = terminated or truncated