#!/usr/bin/env python
# coding: utf-8

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def fit_to_rmse_sinc(fit):
    N = 1000
    u_val = np.linspace(-10, 10, N).reshape(-1, 1)
    y_val = np.sin(u_val) / u_val

    var = np.var(y_val)
    rmse = (1 - fit/100 )*np.sqrt(var)
    return rmse


def main():
    # 学習中の精度と勾配のプロット
    
    date = '2022-09-13'
    FIGURE_FOLDER = './figures/HierDNN/' + date + '/'

    date_list_HierDNN = ['2022-04-25-5-64', '2022-04-25-17-23']
    # date_list_HierDNN = ['2023-01-04-3-5-64', '2022-04-25-17-23']
    date_list_DNN = ['2022-04-25-5-64', '2022-04-25-17-23']
    num_layer = [5, 17]
    hidden_size = [64, 23]

    num_params = 8645


    plt.figure(tight_layout=True, figsize=[6.4, 3.5])
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams["font.family"] = "Times New Roman" 
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["font.size"] = 15

    colormap = ['r', 'g', 'b', 'k']
    linestyle_list = ['-', '--', '-.', ':']
    marker_list = ['o', '^', 'X', 'D']
    for i in range(len(num_layer)):
        # data_DNN = np.loadtxt(f'./figures/DNN/{date_list_DNN[i]}/accuracy.csv', delimiter=',')
        log_DNN = np.loadtxt(f'./figures/DNN/{date_list_DNN[i]}/log.csv', delimiter=',')
        plt.plot(fit_to_rmse_sinc(log_DNN[:, 3]), log_DNN[:, 2]*num_params, linestyle_list[2+i], color=colormap[2+i], zorder=1, alpha=0.5)
        # plt.plot(log_DNN[:, 3], log_DNN[:, 2]*num_params, color=colormap[2+i], zorder=1, alpha=0.4)
        # plt.plot(log_DNN[:, 3], log_DNN[:, 2]*num_params, color=cm.tab20(4*i+1), zorder=1)

        # data_HierDNN = np.loadtxt(f'./figures/HierDNN/{date_list_HierDNN[i]}/accuracy.csv', delimiter=',')
        log_HierDNN = np.loadtxt(f'./figures/HierDNN/{date_list_HierDNN[i]}/log.csv', delimiter=',')
        plt.plot(fit_to_rmse_sinc(log_HierDNN[:, 3]), log_HierDNN[:, 2]*num_params, linestyle_list[i], color=colormap[i], zorder=1, alpha=0.5)
        # plt.plot(log_HierDNN[:, 3], log_HierDNN[:, 2]*num_params, color=cm.tab20(4*i + 3), zorder=1)
    
    for i in range(len(num_layer)):
        log_DNN = np.loadtxt(f'./figures/DNN/{date_list_DNN[i]}/log.csv', delimiter=',')
        plt.scatter(fit_to_rmse_sinc(log_DNN[-1, 3]), log_DNN[-1, 2]*num_params, marker=marker_list[2+i], facecolors='None', color=colormap[2+i], label=f'FNN ({num_layer[i]} layers, {hidden_size[i]} nodes)', zorder=2)
        # plt.scatter(log_DNN[-1, 3], log_DNN[-1, 2]*num_params, color=cm.tab20(4*i), label=f'FNN ({num_layer[i]} layers, {hidden_size[i]} nodes)', zorder=2)

        log_HierDNN = np.loadtxt(f'./figures/HierDNN/{date_list_HierDNN[i]}/log.csv', delimiter=',')
        plt.scatter(fit_to_rmse_sinc(log_HierDNN[-1, 3]), log_HierDNN[-1, 2]*num_params, marker=marker_list[i], facecolors='None', color=colormap[i], label=f'Proposed model ({num_layer[i]} layers, {hidden_size[i]} nodes)', zorder=2)
        # plt.scatter(log_HierDNN[-1, 3], log_HierDNN[-1, 2]*num_params, color=cm.tab20(4*i + 2), label=f'Proposed model ({num_layer[i]} layers, {hidden_size[i]} nodes)', zorder=2)


    plt.grid()
    plt.xlabel('RMSE')
    plt.ylabel('L1 norm of the gradient')
    plt.yscale('log')
    plt.xlim((0, 0.4))
    plt.xticks(np.arange(0, 0.41, 0.1))
    # plt.legend()
    plt.savefig(FIGURE_FOLDER + 'fig_RMSE_vs_grad.png')
    plt.savefig(FIGURE_FOLDER + 'fig_RMSE_vs_grad.pdf')
    plt.show()


if __name__ == '__main__':
    main()