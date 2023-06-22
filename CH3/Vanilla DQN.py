from GridWorld import GridWorld
from Model import FC
import numpy as np
import torch
import random
import matplotlib.pyplot as plt

### HYPERPARAMS
model = FC(input_size=64, hidden_size1=150, hidden_size2=100, output_size=4)
loss_fn = torch.nn.MSELoss()
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
gamma = 0.9
epsilon = 1.0

### TRAINING
action_map = {0: 'u', 1: 'd', 2:'l', 3:'r'}
epochs = int(input("EPOCHS: "))
losses = []
rewards = []

def train_model():

    global epsilon, action_map, epochs, losses, rewards, gamma, optimizer, loss_fn

    for i in range(epochs):                     # EPOCH LOOP
        game = GridWorld(size=4, mode='static')
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

            #print("action:" + action_map[action])

            # Take action, get reward
            #game.display()
            game.makeMove(action_map[action])
            #game.display()
            reward = game.give_reward()
            #print("reward:" + str(reward))
            reward_episode += reward

            if reward != -1: 
                done = True
                rewards.append(reward_episode)
            
            else:
                new_state = game.render_np().reshape(1, 64) + np.random.rand(1,64)/10.0
                new_state = torch.from_numpy(new_state).float() # TO TENSOR, as FLOAT
                state = new_state

                # Find maxQ of new state
                with torch.no_grad():
                    q_new = model(new_state)
                q_max = torch.max(q_new)
            
            Y = reward + gamma * q_max if reward == -1 else reward
            Y = torch.Tensor([Y]).detach().float().squeeze()          # 여기서 Y는 상수(TD-target)이므로 기울기 계산 X
            X = q.squeeze()[action]  
            
            print(f"X is {X} with type {type(X)}")
            print(f"Y is {Y} with type {type(Y)}") 

            # BACKPROPAGATION
            loss = loss_fn(X, Y)
            optimizer.zero_grad()
            loss.backward()
            losses.append(loss.item())               # FOR GRAPHING
            optimizer.step()

            if epsilon > 0.1: epsilon -= (1/epochs)

    return model

def test_model(model):

    print("MODEL PERFORMS A TEST CASE")

    game = GridWorld(size=4, mode='static')
    state = game.render_np().reshape(1, 64) # FLATTENING
    state += np.random.rand(1, 64) / 10.0   # ADD NOISE
    state = torch.from_numpy(state).float() # TO TENSOR, AS FLOAT
    done = False
    reward_episode = 0

    while done is False: # EPISODE LOOP
        q = model(state)
        q_ = q.detach().numpy()   # TO NUMPY
        action = np.argmax(q_)

        print("action:" + action_map[action])

        # Take action, get reward
        game.display()
        game.makeMove(action_map[action])
        game.display()
        reward = game.give_reward()
        print("reward:" + str(reward))
        reward_episode += reward

        if reward != -1: 
            done = True
            break
        
        else:
            new_state = game.render_np().reshape(1, 64) + np.random.rand(1,64)/10.0
            new_state = torch.from_numpy(new_state).float() # TO TENSOR, as FLOAT
            state = new_state


test_model(train_model())
plt.plot(rewards)
plt.show()
