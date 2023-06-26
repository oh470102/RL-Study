### NOTE ::: REINFORCE IS A MC(MONTE-CARLO) ALGORITHM, NOT TD, SO IS UPDATED AFTER EVERY EPISODE

from Model import Model
from utils import *
import numpy as np
import torch
import gym
import matplotlib.pyplot as plt

### RESOLVE MATPLOTLIB ERROR
matplotlib_error()

### HYPERPARAMS
MAX_EPISODES = int(input("EPOCHS(MAX EPISODES)?: "))
MAX_DURATION = int(input("SET DURATION LENGTH TO?: "))
gamma = 0.99
learning_rate = 9e-3

### MODEL
model = Model(input_size=4, hidden_size=150, output_size=2)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = lambda preds, discG: -1 * torch.sum(discG * torch.log(preds))

### TRAINING
scores = []
losses = []
def train_model():
    env = gym.make('CartPole-v1')
    global scores, losses

    for episode in range(MAX_EPISODES):

        print(f"currently on episode {episode}/{MAX_EPISODES}")

        curr_state, info = env.reset()
        done = False
        transitions = []
        i = 0

        while not done:

            i += 1
            if i > MAX_DURATION: break

            action_prob_dist = model(torch.from_numpy(curr_state).float())
            action = np.random.choice(np.array([0,1]), p=action_prob_dist.detach().numpy())
            next_state, reward, done, _, _ = env.step(action)
            transitions.append((curr_state, action, reward))
            curr_state = next_state

            if done: break

        scores.append(len(transitions))

        reward_batch = torch.Tensor([r for (s,a,r) in transitions]) # flip(dim=0) to reverse order, this is done in utils already.
        disc_returns = discount_rewards(reward_batch)
        state_batch = torch.Tensor(np.array([s for (s,a,r) in transitions]))
        action_batch = torch.Tensor([a for (s,a,r) in transitions])
        pred_batch = model(state_batch) # 아직까지 model이 학습하지 않았기 때문에 같은 prob. 이 나옴
        prob_batch = pred_batch.gather(dim=1, index=action_batch.long().view(-1, 1)).squeeze()  # action_batch()를 2차원 배열로 만들고, 세로로 나열하기. 1->2로 됐기 때문에 squeeze 해주고

        loss = loss_fn(prob_batch, disc_returns.detach())
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model, scores

def test_model(model):

    env = gym.make('CartPole-v1', render_mode='human')
    curr_state, _  = env.reset()
    env.render()
    done = False

    for i in range(0, MAX_DURATION):
        action_prob_dist = model(torch.from_numpy(curr_state).float())
        action = np.random.choice(np.array([0,1]), p=action_prob_dist.detach().numpy())
        next_state, reward, done, _, _ = env.step(action)
        curr_state = next_state

        if done: break

trained_model, scores = train_model()
test_model(trained_model)
print(scores)

plt.plot(moving_average(scores, 50, reg=False))
plt.show()
        
plt.plot(moving_average(losses, 50, reg=False))
plt.show()
        