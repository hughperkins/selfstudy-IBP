import matplotlib.pyplot as plt
import numpy as np


zerov = np.array([0, 0])


def plot_vector(start, end, formatstr=None, **kwargs):
    line = np.zeros((2, 2))
    line[0] = start
    line[1] = end
    if formatstr is None:
        plt.plot(line[:, 0], line[:, 1], **kwargs)
    else:
        plt.plot(line[:, 0], line[:, 1], formatstr, **kwargs)
