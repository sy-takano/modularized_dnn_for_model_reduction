#!/usr/bin/env python
# coding: utf-8

import sys
import numpy as np
import matplotlib.pyplot as plt

# 学習中の評価関数の変化を各縮約モデルごとにプロット

# date = '2022-09-03'
# FIGURE_FOLDER = './figures/HierDNN/' + date + '/'

num_layer = [50]
# hidden_size = [64 ,37, 23, 15]

# fig, axes = plt.subplots(nrows=4, ncols=1, sharex=True)
for i in range(len(num_layer)):
    log = np.loadtxt(f'figures/LFR_with_HierLTIandMatrix/2022-11-23/log.csv', delimiter=',')
    # plt.plot(log[:, 0], log[:, 1])
    log = np.concatenate([np.arange(0, 15000).reshape(-1, 1) , log], 1)
    # print(log)
    for jj in range(1, num_layer[i]+1):
        temp = log[log[:, 1] == jj]
        plt.plot(temp[:, 0], temp[:, 2], label=f'l = {jj}')

    plt.yscale('log')
    plt.grid()
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    # plt.xlim(-800, 15800)
    # plt.legend(loc='best')
    # plt.set_title(f'{num_layer[i]} layers, {hidden_size[i]} nodes')

plt.xlabel('Epoch')
# plt.legend()
# plt.savefig(FIGURE_FOLDER + 'fig_loss.png')
plt.show()