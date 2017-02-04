import matplotlib.pyplot as plt
import numpy as np
import math


zerov = np.array([0, 0])


def plot_vector(p2, formatstr=None, p1=zerov, **kwargs):
    # if p1 is None:
    #     # p2 = p1
    #     p1 = zerov
    line = np.zeros((2, 2))
    line[0] = p1
    line[1] = p2
    # print('kwargs', kwargs, 'formatstr', formatstr)
    # plt.quiver(line[:, 0], line[:, 1], color='red', label='foo')
    # plt.arrow(p1[0], p1[1], p2[0], p2[1], color='red', label='foo')
    if formatstr is None:
        plt.plot(line[:, 0], line[:, 1], **kwargs)
    else:
        plt.plot(line[:, 0], line[:, 1], formatstr, **kwargs)


def get_tick_spacing(max_range):
    power10 = int(math.log(max_range) / math.log(10))
    scaled_max_range = max_range * math.pow(10, power10)
    if scaled_max_range < 2.5:
        spacing = math.pow(10, power10) * 0.5
    elif scaled_max_range < 5:
        spacing = math.pow(10, power10)
    else:
        spacing = math.pow(10, power10) * 5
    return spacing


def get_ticks(lim, spacing):
    tick_min = int(lim[0] / spacing) * spacing
    tick_max = (int(lim[1] / spacing) + 1) * spacing
    ticks = np.arange(tick_min, tick_max, spacing)
    return ticks


def proportional_axes(xlim, ylim, size=10):
    x_range = xlim[1] - xlim[0]
    y_range = ylim[1] - ylim[0]
    max_range = max(x_range, y_range)
    tick_spacing = get_tick_spacing(max_range)
    xticks = get_ticks(xlim, tick_spacing)
    yticks = get_ticks(ylim, tick_spacing)
    plt.figure(figsize=(size * x_range / max_range, size * y_range / max_range))
    plt.xlim(xlim[0], xlim[1])
    plt.ylim(ylim[0], ylim[1])
    plt.xticks(xticks)
    plt.yticks(yticks)


def proj_ortho_basis(v, b0, b1):
    v = v.reshape(-1)
    b0 = b0.reshape(-1)
    b1 = b1.reshape(-1)
    x = v.dot(b0)
    y = v.dot(b1)
    proj = np.array([x, y], dtype=np.float32)
    return proj


# def plot_vector(start, end, formatstr=None, **kwargs):
#     line = np.zeros((2, 2))
#     line[0] = start
#     line[1] = end
#     if formatstr is None:
#         plt.plot(line[:, 0], line[:, 1], **kwargs)
#     else:
#         plt.plot(line[:, 0], line[:, 1], formatstr, **kwargs)
