import numpy as np

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