from GridWorld import GridWorld
from Model import FC
from utils import moving_average
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from collections import deque

### HYPERPARAMS
model = FC(input_size=64, hidden_size1=150, hidden_size2=100, output_size=4)
loss_fn = torch.nn.MSELoss()
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
gamma = 0.9
epsilon = 1.0
mem_size = 5000
batch_size = 200
replay = deque(maxlen=mem_size)

### TRAINING
action_map = {0: 'u', 1: 'd', 2:'l', 3:'r'}
epochs = int(input("EPOCHS: "))
losses = []
rewards = []

def train_model():

    global epsilon, action_map, epochs, losses, rewards, gamma, optimizer, loss_fn

    for i in range(epochs):                     # EPOCH LOOP
        print(f"currently on epoch {i}")
        game = GridWorld(size=4, mode='random')
        state = game.render_np().reshape(1, 64) # FLATTENING
        state += np.random.rand(1, 64) / 10.0   # ADD NOISE
        state = torch.from_numpy(state).float() # TO TENSOR, AS FLOAT
        done = False
        reward_episode = 0

        while done is False: # EPISODE LOOP
            q = model(state)          # TENSOR
            q_ = q.detach().numpy()   # NUMPY
            if random.random() < epsilon:       # SELECT ACTION
                action = np.random.randint(0, 4)
            else:
                action = np.argmax(q_)

            # env.step()
            game.makeMove(action_map[action])
            next_state = torch.from_numpy(game.render_np().reshape(1, 64)).float()
            reward = game.give_reward()
            reward_episode += reward
            done = False if reward == -1 else True
            exp = (state, action, reward, next_state, done)
            replay.append(exp)
            state = next_state

            # ER 구현
            if len(replay) > batch_size:
                minibatch = random.sample(replay, batch_size)
                # tensor인 애들은 concatenate, 스칼라인 애들은 Tensor로
                state_batch = torch.cat([s1 for (s1, a, r, s2, d) in minibatch])
                action_batch = torch.Tensor([a for (s1, a, r, s2, d) in minibatch])
                reward_batch = torch.Tensor([r for (s1, a, r, s2, d) in minibatch])
                state2_batch = torch.cat([s2 for (s1, a, r, s2, d) in minibatch])
                done_batch = torch.Tensor([d for (s1, a, r, s2, d) in minibatch])

                print(state_batch.size())
                break

                q = model(state_batch)
                with torch.no_grad():
                    q_next = model(state2_batch)

                Y = reward_batch + gamma * ((1 - done_batch) * torch.max(q_next, dim=1)[0]) # dim ~ axis
                                
                                                                                            # max -> [values, indices]
                X = q.gather(dim=1, index=action_batch.long().unsqueeze(dim=1)).squeeze()   # selects chosen q-values based on actions along axis 1
                                                                                            # .long() for float -> int
                loss = loss_fn(X, Y.detach())                                               # unsqueeze/squeeze to match shape of q and index tensor: [tensor], [tensor]
                optimizer.zero_grad()
                loss.backward()
                losses.append(loss.item())
                optimizer.step()     

            if done: rewards.append(reward_episode)
            if epsilon > 0.1: epsilon -= (1/epochs)

    return model

def test_model(model):

    print("MODEL PERFORMS A TEST CASE")

    game = GridWorld(size=4, mode='random')
    state = game.render_np().reshape(1, 64) # FLATTENING
    state += np.random.rand(1, 64) / 10.0   # ADD NOISE
    state = torch.from_numpy(state).float() # TO TENSOR, AS FLOAT
    done = False
    reward_episode = 0

    while done is False: # EPISODE LOOP
        q = model(state)
        q_ = q.detach().numpy()   # TO NUMPY
        action = np.argmax(q_)

        # Take action, get reward
        game.display()
        game.makeMove(action_map[action])
        game.display()
        reward = game.give_reward()
        reward_episode += reward

        if reward != -1: 
            done = True
            break
        
        else:
            new_state = game.render_np().reshape(1, 64) + np.random.rand(1,64)/10.0
            new_state = torch.from_numpy(new_state).float() # TO TENSOR, as FLOAT
            state = new_state

test_model(train_model())
plt.plot(moving_average(array=rewards, window_size=10))
#plt.plot(moving_average(array=losses, window_size=50))
plt.show()