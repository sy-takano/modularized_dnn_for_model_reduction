#!/usr/bin/env python
# coding: utf-8

import sys
import numpy as np
import matplotlib.pyplot as plt


# 学習中のイタレーションに対する勾配をプロット

date = '2022-04-20'
FIGURE_FOLDER = './figures/DNN/' + date + '/'

date_list_DNN = ['2022-04-25', '2022-04-25']
date_list_HierDNN = ['2022-04-25', '2022-04-25']
num_layer = [5, 17]
hidden_size = [64, 23]

plt.figure()
for i in range(len(num_layer)):
    log = np.loadtxt(f'./figures/DNN/{date_list_DNN[i]}-{num_layer[i]}-{hidden_size[i]}/log.csv', delimiter=',')
    plt.plot(log[:, 0], log[:, 2], '-.', label=f'FNN ({num_layer[i]} layers, {hidden_size[i]} nodes)')

    log = np.loadtxt(f'./figures/HierDNN/{date_list_HierDNN[i]}-{num_layer[i]}-{hidden_size[i]}/log.csv', delimiter=',')
    plt.plot(log[:, 0], log[:, 2], '--', label=f'Proposed model ({num_layer[i]} layers, {hidden_size[i]} nodes)')



plt.yscale('log')
plt.grid()
plt.ylabel('Mean of absolute value of each element in gradient')
plt.xlabel('Epoch')
plt.legend()
plt.savefig(FIGURE_FOLDER + 'fig_grad_compare.png')
plt.show()
