# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 16:25:23 2024

@author: 39829
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 定义图形和轴
fig, ax = plt.subplots()
xdata, ydata = np.linspace(0, 2*np.pi, 100), []
ln, = plt.plot([], [], 'r', animated=True)

# 初始化函数：图形的背景
def init():
    ax.set_xlim(0, 2*np.pi)
    ax.set_ylim(-1, 1)
    return ln,

# 更新函数：对每帧调用以更新图形
def update(frame):
    ydata = np.sin(xdata + frame / 10)
    ln.set_data(xdata, ydata)
    return ln,

# 创建动画
ani = FuncAnimation(fig, update, frames=np.arange(0, 100, 1), init_func=init, blit=True)

# 保存动画为GIF
ani.save('sine_wave_animation.gif', writer='pillow', fps=30)
