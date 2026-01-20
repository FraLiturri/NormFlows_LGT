# Status: done ✅
import numpy as np


def add_metrics(history, mtrcs): #updates a dictionary; 
    for key, val in mtrcs.items():
        history[key].append(val)


def average_metrics(history, avg_last_N_epochs, keys):
    avg = {}
    for key in keys:
        if history[key]: #checks if the key is in the dictionary; 
            avg_val = np.mean(history[key][-avg_last_N_epochs:]) #calculates the mean for the last N epochs ([-i:] returns the last i elements); 
            avg[key] = avg_val
        else:
            avg[key] = np.nan

    return avg


def print_dict(dct, pre="", **kwargs): #prints the dictionary elemntwise; 
    for key, val in dct.items():
        print(f"{pre}{key} {dct[key]:g}", **kwargs)


def dict_to_numpy(dct, keys): #conversion dict->numpy; 
    array = []
    for key in keys:
        array.append(np.asarray(dct[key]))
    return np.stack(array, 1)


def average_history(history, n):
    """
    Averages the history of metrics over every n entries.

    Parameters:
    history (numpy array): A 2D array of shape (n, m) where n is the number of entries and m is the number of metrics.
    n (int): The number of entries to average over.

    Returns:
    list: A list of averaged metrics, including the epochs and time.
    """
    kernel = np.ones((n,)) / n
    n_cols = history.shape[1]
    avgs = []
    for i in range(1, n_cols):
        avgs.append(np.convolve(history[:, i], kernel, mode="valid"))
    n_avgs_rows = len(avgs[0])
    time = history[n // 2 : n // 2 + n_avgs_rows, 0]
    epochs = np.arange(n // 2, n // 2 + n_avgs_rows)
    avgs = [epochs] + [time] + avgs
    return avgs
