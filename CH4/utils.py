import numpy as np
import torch

def matplotlib_error():
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def moving_average(array, window_size, reg=None):
    moving_averages = []
    for i in range(len(array) - window_size + 1):
        window = array[i:i+window_size]
        average = np.mean(window)
        moving_averages.append(average)

    if reg is True:
        return [i/np.max(moving_averages) for i in moving_averages]
    else:
        return moving_averages
    
def discount_rewards(rewards, gamma=0.99):
    G = 0
    returns = []
    for i in range(len(rewards)-1, -1, -1):
        G = rewards[i] + gamma*G
        returns.insert(0, G)

    # Normalize the returns (optional)
    returns = torch.tensor(returns)
    returns = returns / torch.max(returns) # 이상하게 표준화를 하면 softmax에서 NaN 발생...

    return returns
        
