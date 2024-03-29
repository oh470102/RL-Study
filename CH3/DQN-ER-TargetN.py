from GridWorld import GridWorld
from Model import FC
from utils import moving_average
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from collections import deque
import copy

### HYPERPARAMS
loss_fn = torch.nn.MSELoss()
learning_rate = 1e-3
gamma = 0.91
epsilon = 1.0
mem_size = 5000
batch_size = 200
synch_freq = 500
synch_i = 0

### MODEL 
model = FC(input_size=64, hidden_size1=150, hidden_size2=100, output_size=4)
target_model = copy.deepcopy(model)
target_model.load_state_dict(model.state_dict()) # INIT. EQUALIZES PARAMS
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
replay = deque(maxlen=mem_size)

### TRAINING
losses = []
rewards = []
action_map = {0: 'u', 1: 'd', 2:'l', 3:'r'}
epochs = int(input("EPOCHS: "))

def train_model():

    global epsilon, action_map, epochs, losses, rewards, gamma, optimizer, loss_fn, synch_i

    for i in range(epochs):                     # EPOCH LOOP
        print(f"currently on epoch {i}")
        game = GridWorld(size=4, mode='random')
        state = game.render_np().reshape(1, 64) # FLATTENING
        state += np.random.rand(1, 64) / 10.0   # ADD NOISE
        state = torch.from_numpy(state).float() # TO TENSOR, AS FLOAT
        done = False
        reward_episode = 0

        while done is False: # EPISODE LOOP

            synch_i += 1
            if synch_i % synch_freq == 0:
                target_model.load_state_dict(model.state_dict()) # 동기화. COUNT EVERY STEP!

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

                q = model(state_batch)
                with torch.no_grad():
                    q_next = target_model(state2_batch) # 다음 state의 Q값은 목표먕으로 예측

                Y = reward_batch + gamma * ((1 - done_batch) * torch.max(q_next, dim=1)[0]) # dim ~ axis
                                
                                                                                            # max -> [values, indices]
                X = q.gather(dim=1, index=action_batch.long().unsqueeze(dim=1)).squeeze()   # selects chosen q-values based on actions along axis 1
                                                                                            # .long() for float -> int
                loss = loss_fn(X, Y.detach())                                               # unsqueeze/squeeze to match shape of q and index tensor: [tensor], [tensor]
                optimizer.zero_grad()
                loss.backward()
                losses.append(loss.item())
                optimizer.step()      # 여기서 optimizer의 model param에 따라 학습되는 모델이 갈림.
                                      # update되는 parameters은 model 이라는 점 유의! (target model X)
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
plt.plot(moving_average(array=rewards, window_size=10, reg=True))
plt.plot(moving_average(array=losses, window_size=50, reg=True))
plt.show()