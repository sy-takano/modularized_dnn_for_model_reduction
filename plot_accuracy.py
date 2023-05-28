#!/usr/bin/env python
# coding: utf-8

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# 学習後にモデルを縮約した時の精度をプロット

def main_average():
    date = '2022-01-11'
    FIGURE_FOLDER = './figures/HierDNN/' + date + '/'

    num_layer = [4, 8, 16, 32] + 1
    hidden_size = [64 ,37, 23, 15]
    num_iter = 10

    data_DNN = np.loadtxt(f'./figures/DNN/2021-12-16/accuracy.csv', delimiter=',')
    # data_DNN = np.loadtxt(f'./figures/DNN/{date}/accuracy.csv', delimiter=',')


    plt.figure(tight_layout=True)
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams["font.family"] = "Times New Roman" 
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["font.size"] = 15

    plt.plot(data_DNN[:, 1], data_DNN[:, 2], 'b', marker='o', markerfacecolor='None',label='Pruned FNN (5 layers, 64 nodes)')

    # colormap = plt.get_cmap('tab20c')
    colormap = ['red', 'k', 'm', 'orange']
    linestyle_list = ['-', '-', '-', '-']
    marker_list = ['X', '<', 'D']

    # for i in range(len(num_layer)):
    for i in [0, 2]:
        data_HierDNN = []
        for jj in range(num_iter):
            data_HierDNN.append(np.loadtxt(f'./figures/HierDNN/{date}-{num_layer[i]}-{hidden_size[i]}/accuracy_{jj}.csv', delimiter=','))
        
        data = np.stack(data_HierDNN, 0)
        q50 = np.array([])
        q75 = np.array([])
        q25 = np.array([])
        q0 = np.array([])
        q100 = np.array([])
        mean = np.array([])
        for kk in range(num_layer[i]):
            q50_temp, q75_temp, q25_temp, q0_temp, q100_temp = np.percentile(data[:, kk, 2], [50, 75, 25, 0, 100])
            q50 = np.append(q50, q50_temp)
            q75 = np.append(q75, q75_temp)
            q25 = np.append(q25, q25_temp)
            q0 = np.append(q0, q0_temp)
            q100 = np.append(q100, q100_temp)

            mean = np.append(mean, np.mean(data[:, kk, 2]))
        
        # plt.plot(data[0, :, 1], q25, '-.', color=colormap(4*i+2), marker='o')
        # plt.plot(data[0, :, 1], q75, '-.', color=colormap(4*i+2), marker='o')
        # plt.plot(data[0, :, 1], q0, '-.', color=colormap(4*i+3), marker='o')
        # plt.plot(data[0, :, 1], q100, '-.', color=colormap(4*i+3), marker='o')
        # plt.plot(data[0, :, 1], q50, '-.', color=colormap(4*i), marker='o', label=f'Proposed model ({num_layer[i]} layers, {hidden_size[i]} nodes)')

        plt.plot(data[0, :, 1], mean, linestyle_list[i], color=colormap[i], marker=marker_list[i], markerfacecolor='None', label=f'Proposed model ({num_layer[i]+1} layers, {hidden_size[i]} nodes)')



    plt.grid()
    plt.xlabel('Number of parameters')
    plt.ylabel('Fit rate [%]')
    plt.xscale('log')
    # plt.legend()
    plt.savefig(FIGURE_FOLDER + 'fig_accuracy.png')
    plt.show()


def main():
    date = '2022-09-13'
    FIGURE_FOLDER = './figures/HierDNN/' + date + '/'

    date_list_HierDNN = ['2022-04-25-5-64', '2022-04-25-17-23']
    date_list_DNN = ['2022-04-25-5-64', '2022-04-25-17-23']
    num_layer = [5, 17]
    hidden_size = [64, 23]
    num_iter = 1

    # data_DNN = np.loadtxt(f'./figures/DNN/{date}/accuracy.csv', delimiter=',')


    plt.figure(tight_layout=True)
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams["font.family"] = "Times New Roman" 
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["font.size"] = 15
    # plt.plot(data_DNN[:, 1], data_DNN[:, 2], 'b', marker='o', markerfacecolor='None', label='Pruned FNN (5 layers, 64 nodes)')

    colormap = ['r', 'b']
    linestyle_list = ['-', '-']
    marker_list = ['X', 'D']

    for i in range(len(num_layer)):
        for jj in range(num_iter):
            data_HierDNN = np.loadtxt(f'./figures/HierDNN/{date_list_HierDNN[i]}/accuracy.csv', delimiter=',')
            data_DNN = np.loadtxt(f'./figures/DNN/{date_list_DNN[i]}/accuracy.csv', delimiter=',')
            if jj == 0:
                plt.plot(data_HierDNN[:, 1], data_HierDNN[:, 2], linestyle_list[i], color=colormap[0], marker=marker_list[i], markerfacecolor='None', label=f'Proposed model ({num_layer[i]} layers, {hidden_size[i]} nodes)')
                plt.plot(data_DNN[:, 1], data_DNN[:, 2], linestyle_list[i], color=colormap[1], marker=marker_list[i], markerfacecolor='None', label=f'FNN ({num_layer[i]} layers, {hidden_size[i]} nodes)')
            else:
                plt.plot(data_HierDNN[:, 1], data_HierDNN[:, 2],linestyle_list[i], color=colormap[i], marker=marker_list[i], markerfacecolor='None')

    plt.grid()
    plt.xlabel('Number of parameters')
    plt.ylabel('Fit rate [%]')
    plt.xscale('log')
    # plt.legend()
    plt.savefig(FIGURE_FOLDER + 'fig_accuracy.png')
    plt.show()

def main2():
    # date = '2022-04-25-17-23'
    date = '2022-09-21-california'
    FIGURE_FOLDER = './figures/HierDNN/' + date + '/'

    # date_list_HierDNN = [f'{date}/retrained']
    # date_list_DNN = [f'{date}/retrained']
    date_list_HierDNN = ['2022-09-21-5-64-california/retrained']
    date_list_DNN = ['2022-09-21-5-64-california/retrained']

    num_layer = [17]
    hidden_size = [23]
    num_iter = 1

    fit_or_rmse = 1 # fitの場合は0, RMSEの場合は1

    plt.figure(tight_layout=True, figsize=[6.4, 3.5])
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams["font.family"] = "Times New Roman" 
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["font.size"] = 15

    colormap = ['r', 'm', 'b', 'c']
    linestyle_list = ['-', '--', '-.', ':']
    marker_list = ['o', '^', 'X', 'D']

    for i in range(len(num_layer)):
        data_HierDNN = np.loadtxt(f'./figures/HierDNN/{date_list_HierDNN[i]}/accuracy.csv', delimiter=',')
        data_DNN = np.loadtxt(f'./figures/DNN/{date_list_DNN[i]}/accuracy.csv', delimiter=',')
        
        plt.plot(data_DNN[:, 1], data_DNN[:, 2+fit_or_rmse], linestyle_list[3], color=colormap[3], marker=marker_list[3], markerfacecolor='None', label=f'FNN (retrained)')
        plt.plot(data_DNN[:, 1], data_DNN[:, 4+fit_or_rmse], linestyle_list[2], color=colormap[2], marker=marker_list[2], markerfacecolor='None', label=f'FNN (not retrained)')
        plt.plot(data_HierDNN[:, 1], data_HierDNN[:, 2+fit_or_rmse], linestyle_list[1], color=colormap[1], marker=marker_list[1], markerfacecolor='None', label=f'Proposed model (retrained)')
        plt.plot(data_HierDNN[:, 1], data_HierDNN[:, 4+fit_or_rmse], linestyle_list[0], color=colormap[0], marker=marker_list[0], markerfacecolor='None', label=f'Proposed model (not retrained)')

    plt.grid()
    plt.xscale('log')
    # plt.legend()
    plt.xlabel('Number of parameters')

    if fit_or_rmse == 0:
        plt.ylabel('Fit rate [%]')
        
        plt.savefig(FIGURE_FOLDER + 'fig_accuracy_retrained.png')
        plt.savefig(FIGURE_FOLDER + 'fig_accuracy_retrained.pdf')
    elif fit_or_rmse == 1:
        plt.ylabel('RMSE')
        plt.savefig(FIGURE_FOLDER + 'fig_RMSE_retrained.png')
        plt.savefig(FIGURE_FOLDER + 'fig_RMSE_retrained.pdf')
    
    plt.show()

def main3():
    # FIGURE_FOLDER = './figures/HierDNN/2023-01-04-5-64/'
    # date_list_HierDNN = ['2022-04-25-5-64']
    # date_list_DNN = ['2023-01-04-5-64']    

    # FIGURE_FOLDER = './figures/HierDNN/2023-01-05-17-23/'
    # date_list_HierDNN = ['2022-04-24-17-23']
    # date_list_DNN = ['2023-01-05-17-23']

    # FIGURE_FOLDER = './figures/HierDNN/2023-01-05-5-64-california/'
    # date_list_HierDNN = ['2022-09-21-5-64-california/retrained']
    # date_list_DNN = ['2023-01-05-5-64-california']

    # FIGURE_FOLDER = './figures/HierDNN/2023-01-05-17-23-california/'
    # date_list_HierDNN = ['2022-09-21-17-23-california/retrained']
    # date_list_DNN = ['2023-01-05-17-23-california']

    # FIGURE_FOLDER = './figures/HierDNN/2023-01-08-5-64/'
    # date_list_HierDNN = ['2022-04-25-5-64']
    # date_list_DNN = ['2023-01-08-5-64']    

    # FIGURE_FOLDER = './figures/HierDNN/2023-01-08-17-23/'
    # date_list_HierDNN = ['2022-04-24-17-23']
    # date_list_DNN = ['2023-01-08-17-23']

    # FIGURE_FOLDER = './figures/HierDNN/2023-01-08-5-64-california/'
    # date_list_HierDNN = ['2022-09-21-5-64-california/retrained']
    # date_list_DNN = ['2023-01-08-5-64-california']

    # FIGURE_FOLDER = './figures/HierDNN/2023-01-08-17-23-california/'
    # date_list_HierDNN = ['2022-09-21-17-23-california/retrained']
    # date_list_DNN = ['2023-01-08-17-23-california']

    FIGURE_FOLDER = './figures/HierDNN/2023-01-09-5-64/'
    date_list_HierDNN = ['2022-04-25-5-64']
    date_list_DNN = ['2023-01-09-5-64']

    num_layer = [17]
    # hidden_size = [64]
    num_iter = 1

    plt.figure(tight_layout=True)
    # plt.figure(tight_layout=True, figsize=[6.4, 3.5])
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams["font.family"] = "Times New Roman" 
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["font.size"] = 15

    colormap = ['r', 'g', 'b', 'k']
    linestyle_list = ['-', '--', '-.', ':']
    marker_list = ['o', '^', 'X', 'D']


    for i in range(len(num_layer)):
        for jj in range(num_iter):
            data_HierDNN = np.loadtxt(f'./figures/HierDNN/{date_list_HierDNN[i]}/accuracy.csv', delimiter=',')
            data_DNN = np.loadtxt(f'./figures/HierDNN/{date_list_DNN[i]}/accuracy.csv', delimiter=',')
            if jj == 0:
                # plt.plot(data_HierDNN[:, 1], data_HierDNN[:, 3], linestyle_list[i], color=colormap[0], marker=marker_list[1], markerfacecolor='None', label=f'Proposed model ({num_layer[i]} layers, {hidden_size[i]} nodes)')
                # plt.plot(data_DNN[:, 1], data_DNN[:, 3], linestyle_list[i], color=colormap[1], marker=marker_list[0], markerfacecolor='None', label=f'FNN ({num_layer[i]} layers, {hidden_size[i]} nodes)')
                plt.plot(data_DNN[:, 1], data_DNN[:, 3], linestyle_list[2+i], color=colormap[2+i], marker=marker_list[2+i], markerfacecolor='None', label=f'Residual learning')
                plt.plot(data_HierDNN[:, 1], data_HierDNN[:, 3], linestyle_list[i], color=colormap[i], marker=marker_list[i], markerfacecolor='None', label=f'Proposed method')

    plt.grid()
    plt.xlabel('Number of parameters')
    plt.ylabel('RMSE')
    plt.xscale('log')
    plt.legend()
    # plt.savefig(FIGURE_FOLDER + 'fig_comparison_RMSE.png')
    # plt.savefig(FIGURE_FOLDER + 'fig_comparison_RMSE.pdf')
    plt.show()

def main4():
    date = '2022-09-13'
    # date = '2022-09-21-california'
    FIGURE_FOLDER = './figures/HierDNN/' + date + '/'

    date_list_HierDNN = ['2022-04-25-5-64/retrained', '2022-04-25-17-23/retrained']
    date_list_DNN = ['2022-04-25-5-64/retrained', '2022-04-25-17-23/retrained']
    # date_list_HierDNN = ['2022-09-21-5-64-california/retrained', '2022-09-21-17-23-california/retrained']
    # date_list_DNN = ['2022-09-21-5-64-california/retrained', '2022-09-21-17-23-california/retrained']

    num_layer = [5, 17]
    hidden_size = [64, 23]
    num_iter = 1

    fit_or_rmse = 1 # fitの場合は0, RMSEの場合は1
    decimate_flag = 0   # 17層モデルの結果をすべて表示せず一部間引く場合は1

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
        data_HierDNN = np.loadtxt(f'./figures/HierDNN/{date_list_HierDNN[i]}/accuracy.csv', delimiter=',')
        data_DNN = np.loadtxt(f'./figures/DNN/{date_list_DNN[i]}/accuracy.csv', delimiter=',')
        
        if decimate_flag == 1 and num_layer[i] == 17:
            data_HierDNN = np.delete(data_HierDNN, [2*jj + 1 for jj in range(7)], 0)
            data_DNN = np.delete(data_DNN, [2*jj + 1 for jj in range(7)], 0)

        plt.plot(data_DNN[:, 1], data_DNN[:, 4+fit_or_rmse], linestyle_list[2+i], color=colormap[2+i], marker=marker_list[2+i], markerfacecolor='None', label=f'FNN (not retrained)')
        plt.plot(data_HierDNN[:, 1], data_HierDNN[:, 4+fit_or_rmse], linestyle_list[i], color=colormap[i], marker=marker_list[i], markerfacecolor='None', label=f'Proposed model (not retrained)')

    plt.grid()
    plt.xscale('log')
    # plt.legend()
    plt.xlabel('Number of parameters')

    if fit_or_rmse == 0:
        plt.ylabel('Fit rate [%]')
        if decimate_flag == 1:
            plt.savefig(FIGURE_FOLDER + 'fig_accuracy_deep_decimated.png')
            plt.savefig(FIGURE_FOLDER + 'fig_accuracy_deep_decimated.pdf')
        else:
            plt.savefig(FIGURE_FOLDER + 'fig_accuracy_deep.png')
            plt.savefig(FIGURE_FOLDER + 'fig_accuracy_deep.pdf')
    elif fit_or_rmse == 1:
        plt.ylabel('RMSE')
        if decimate_flag == 1:
            plt.savefig(FIGURE_FOLDER + 'fig_RMSE_deep_decimated.png')
            plt.savefig(FIGURE_FOLDER + 'fig_RMSE_deep_decimated.pdf')
        else:
            plt.savefig(FIGURE_FOLDER + 'fig_RMSE_deep.png')
            plt.savefig(FIGURE_FOLDER + 'fig_RMSE_deep.pdf')
    plt.show()

def main5():
    # FIGURE_FOLDER = './figures/HierDNN/2023-01-04-5-64/'
    # date_list_HierDNN = ['2022-04-25-5-64']
    # date_list_DNN = ['2023-01-04-5-64']    

    # FIGURE_FOLDER = './figures/HierDNN/2023-01-05-17-23/'
    # date_list_HierDNN = ['2022-04-24-17-23']
    # date_list_DNN = ['2023-01-05-17-23']

    # FIGURE_FOLDER = './figures/HierDNN/2023-01-05-5-64-california/'
    # date_list_HierDNN = ['2022-09-21-5-64-california']

    FIGURE_FOLDER = './figures/HierDNN/2023-01-05-17-23-california/'
    date_list_HierDNN = ['2022-09-21-17-23-california']

    # FIGURE_FOLDER = './figures/HierDNN/2023-01-08-5-64/'
    # date_list_HierDNN = ['2022-04-25-5-64']
    # date_list_DNN = ['2023-01-08-5-64']    

    # FIGURE_FOLDER = './figures/HierDNN/2023-01-08-17-23/'
    # date_list_HierDNN = ['2022-04-24-17-23']
    # date_list_DNN = ['2023-01-08-17-23']

    # FIGURE_FOLDER = './figures/HierDNN/2023-01-08-5-64-california/'
    # date_list_HierDNN = ['2022-09-21-5-64-california/retrained']
    # date_list_DNN = ['2023-01-08-5-64-california']

    # FIGURE_FOLDER = './figures/HierDNN/2023-01-08-17-23-california/'
    # date_list_HierDNN = ['2022-09-21-17-23-california/retrained']
    # date_list_DNN = ['2023-01-08-17-23-california']

    # FIGURE_FOLDER = './figures/HierDNN/2023-01-09-5-64/'
    # date_list_HierDNN = ['2022-04-25-5-64']
    # date_list_DNN = ['2023-01-09-5-64']

    num_layer = [17]
    # hidden_size = [64]
    num_iter = 1

    plt.figure(tight_layout=True)
    # plt.figure(tight_layout=True, figsize=[6.4, 3.5])
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams["font.family"] = "Times New Roman" 
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["font.size"] = 15

    colormap = ['r', 'g', 'b', 'k']
    linestyle_list = ['-', '--', '-.', ':']
    marker_list = ['o', '^', 'X', 'D']


    for i in range(len(num_layer)):
        for jj in range(num_iter):
            data_HierDNN_train = np.loadtxt(f'./figures/HierDNN/{date_list_HierDNN[i]}/accuracy_train.csv', delimiter=',')
            data_HierDNN_val = np.loadtxt(f'./figures/HierDNN/{date_list_HierDNN[i]}/retrained/accuracy.csv', delimiter=',')
            if jj == 0:
                plt.plot(data_HierDNN_train[:, 1], data_HierDNN_train[:, 3], linestyle_list[2+i], color=colormap[2+i], marker=marker_list[2+i], markerfacecolor='None', label=f'Traing data')
                plt.plot(data_HierDNN_val[:, 1], data_HierDNN_val[:, 3], linestyle_list[i], color=colormap[i], marker=marker_list[i], markerfacecolor='None', label=f'Validation data')

    plt.grid()
    plt.xlabel('Number of parameters')
    plt.ylabel('RMSE')
    plt.xscale('log')
    plt.legend()
    # plt.savefig(FIGURE_FOLDER + 'fig_comparison_RMSE_overfitting.png')
    # plt.savefig(FIGURE_FOLDER + 'fig_comparison_RMSE_overfitting.pdf')
    plt.show()

def main_HierLTIandHierDNN():
    date = '2022-12-13'
    # date = '2023-02-05-WHB_zwsize9'
    FIGURE_FOLDER = './figures/LFR_with_HierLTIandHierDNN/' + date + '/'

    data = np.loadtxt(f'{FIGURE_FOLDER}accuracy.csv', delimiter=',')

    # モジュール数を変えた時の精度を3次元プロット
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection' : '3d'})
    
    l_LTI_max = max(data[:, 4]).astype('int32')
    l_NN_max = max(data[:, 5]).astype('int32')
    z = data[:, 3].reshape(l_LTI_max, l_NN_max)

    l_LTI, l_NN = np.meshgrid(np.arange(1, l_LTI_max+1), np.arange(1, l_NN_max+1))

    ax.plot_surface(l_NN, l_LTI, z.transpose(), cmap='plasma')
    # ax.plot_surface(l_LTI, l_NN, z, cmap=plt.cm.Blues)

    ax.scatter(l_NN, l_LTI, z.transpose())
    # ax.scatter(l_LTI, l_NN, z)

    ax.set_ylabel('Number of LTI modules $n$')
    ax.set_xlabel('Number of DNN modules $m$')
    # ax.set_xlabel('Number of LTI modules $n$')
    # ax.set_ylabel('Number of DNN modules $m$')

    ax.set_zlabel(r'RMSE ($\times 10^{-3}$)')

    ax.set_xticks([1, 2, 3, 4])
    ax.set_yticks([1, 2, 3, 4])
    ax.set_zticks([7.16e-3, 7.17e-3, 7.18e-3, 7.19e-3])
    ax.set_zticklabels([7.16, 7.17, 7.18, 7.19])

    plt.savefig(FIGURE_FOLDER + 'fig_RMSE.png')
    plt.savefig(FIGURE_FOLDER + 'fig_RMSE.pdf')


    # モジュール数を変えた時の精度を2次元プロット
    # l_LTI_max = max(data[:, 4]).astype('int32')
    # l_LTI = np.arange(1, l_LTI_max+1)

    # l_NN_max = max(data[:, 5]).astype('int32')
    # l_NN = np.arange(1, l_NN_max+1)

    # rmse = data[:, 3].reshape(l_LTI_max, l_NN_max)

    # colormap = plt.get_cmap('plasma')
    # linestyle_list = ['-', '--', '-.', ':']
    # marker_list = ['o', '^', 'X', 'D']

    # plt.figure()
    # for n in l_LTI:
    #     plt.plot(l_NN, rmse[n-1, :], linestyle_list[n-1], marker=marker_list[n-1], color=colormap((l_LTI_max - n)/l_LTI_max), markerfacecolor='None', label=f'n = {n}')

    # # plt.scatter(data[:, 6], data[:, 3])
    
    # plt.grid()
    # plt.xlabel('$m$')
    # plt.ylabel('RMSE')
    # # plt.xticks([1, 2, 3, 4])

    # plt.savefig(FIGURE_FOLDER + 'fig_RMSE_2D.png')
    # plt.savefig(FIGURE_FOLDER + 'fig_RMSE_2D.pdf')

    plt.show()
    

def main_LTI():
    date = '2022-05-30'
    FIGURE_FOLDER = './figures/HierLTI/' + date + '/'

    date_list = ['2022-05-30-5', '2022-05-26-10', '2022-05-27-20', '2022-05-30-30', '2022-05-27-40']
    num_layer = [5, 10, 20, 30, 40]

    plt.figure(tight_layout=True)
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams["font.family"] = "Times New Roman" 
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["font.size"] = 15

    marker_list = [',', 'o', 'X', 'D', 's']

    for i in range(len(num_layer)):
            data = np.loadtxt(f'./figures/HierLTI/{date_list[i]}/accuracy.csv', delimiter=',')
            if num_layer[i] == 1:
                plt.scatter(data[1], data[2], marker=marker_list[i], label=f'{num_layer[i]} layers')
            else:
                plt.plot(data[:, 1], data[:, 2], color=cm.tab10(i+1), marker=marker_list[i], label=f'{num_layer[i]} layers')

    plt.grid()
    plt.xlabel('Number of parameters')
    plt.ylabel('Fit rate [%]')
    plt.xscale('log')
    # plt.legend()
    plt.savefig(FIGURE_FOLDER + 'fig_accuracy.png')


    order_x = [2*(i+1) for i in range(max(num_layer))]
    plt.figure(tight_layout=True)
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams["font.family"] = "Times New Roman" 
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["font.size"] = 15

    for i in range(len(num_layer)):
            data = np.loadtxt(f'./figures/HierLTI/{date_list[i]}/accuracy.csv', delimiter=',')
            if num_layer[i] == 1:
                plt.scatter(order_x[:num_layer[i]], data[2], marker=marker_list[i], label=f'{num_layer[i]} layers')
            else:
                plt.plot(order_x[:num_layer[i]], data[:, 2], color=cm.tab10(i+1), marker=marker_list[i], label=f'{num_layer[i]} layers')

    plt.grid()
    plt.xlabel('Order')
    plt.ylabel('Fit rate [%]')
    plt.xticks(order_x)
    # plt.xscale('log')
    # plt.legend()
    plt.savefig(FIGURE_FOLDER + 'fig_accuracy_2.png')


    plt.figure(tight_layout=True)
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams["font.family"] = "Times New Roman" 
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["font.size"] = 15

    for i in range(len(num_layer)):
            data = np.loadtxt(f'./figures/HierLTI/{date_list[i]}/accuracy.csv', delimiter=',')
            if num_layer[i] == 1:
                plt.scatter(order_x[:num_layer[i]], data[3], marker=marker_list[i], label=f'{num_layer[i]} layers')
            else:
                plt.plot(order_x[:num_layer[i]], data[:, 3], color=cm.tab10(i+1), marker=marker_list[i], label=f'{num_layer[i]} layers')

    plt.grid()
    plt.xlabel('Order')
    plt.ylabel('RMSE')
    plt.xticks(order_x)
    # plt.xscale('log')
    # plt.legend()
    plt.savefig(FIGURE_FOLDER + 'fig_accuracy_3.png')

    plt.show()


def main6():
    num_layer = [5]
    # hidden_size = [64]
    num_iter = 10

    # FIGURE_FOLDER = './figures/HierDNN/residuals/sinc/'
    FIGURE_FOLDER = './figures/HierDNN/colds/sinc/'
    date_list_HierDNN = [f'proposed/sinc/2023-01-24-{jj}-5-64' for jj in range(num_iter)]
    # date_list_HierDNN2 = [f'residuals/sinc/2023-01-18-{jj}-5-64' for jj in range(num_iter)]
    date_list_HierDNN2 = [f'colds/sinc/2023-01-18-{jj}-5-64-cold' for jj in range(num_iter)]

    # # FIGURE_FOLDER = './figures/HierDNN/2023-01-05-17-23/'
    # date_list_HierDNN = ['2022-04-24-17-23']
    # # date_list_HierDNN2 = [f'residuals/sinc/2023-01-18-{jj}-17-23' for jj in range(num_iter)]
    # date_list_HierDNN2 = [f'colds/sinc/2023-01-18-{jj}-17-23-cold' for jj in range(num_iter)]

    # plt.figure(tight_layout=True)
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
        rmse_hierDNN = []
        rmse_hierDNN2 = []
        for jj in range(num_iter):
            data_HierDNN = np.loadtxt(f'./figures/HierDNN/{date_list_HierDNN[jj]}/accuracy.csv', delimiter=',')
            rmse_hierDNN.append(data_HierDNN[:, 3])

            data_HierDNN2 = np.loadtxt(f'./figures/HierDNN/{date_list_HierDNN2[jj]}/accuracy.csv', delimiter=',')
            rmse_hierDNN2.append(data_HierDNN2[:, 3])
        
        rmse_hierDNN = np.array(rmse_hierDNN)
        rmse_hierDNN2 = np.array(rmse_hierDNN2)
        print(np.mean(rmse_hierDNN, axis=0))
        print(np.mean(rmse_hierDNN2, axis=0))

        # plt.errorbar(x=data_HierDNN[:, 1], y=np.mean(rmse_hierDNN, axis=0), yerr=np.std(rmse_hierDNN, axis=0), capsize=4, fmt='-o', markerfacecolor='None', ecolor='red', color='red', label=f'Residual learning')
        # plt.errorbar(x=data_HierDNN2[:, 1], y=np.mean(rmse_hierDNN2, axis=0), yerr=np.std(rmse_hierDNN2, axis=0), capsize=4, fmt='-.X', markerfacecolor='None', ecolor='blue', color='blue', label=f'Residual learning')

        # data_HierDNN = np.loadtxt(f'./figures/HierDNN/{date_list_HierDNN[0]}/accuracy.csv', delimiter=',')
        # plt.plot(data_HierDNN[:, 1], data_HierDNN[:, 3], linestyle_list[i], color=colormap[i], marker=marker_list[i], markerfacecolor='None', label=f'Proposed method', alpha=0.3)

        for jj in range(num_iter):
            data_HierDNN2 = np.loadtxt(f'./figures/HierDNN/{date_list_HierDNN2[jj]}/accuracy.csv', delimiter=',')
            # data_HierDNN = np.loadtxt(f'./figures/HierDNN/{date_list_HierDNN[jj]}/accuracy.csv', delimiter=',')

            if jj == 0:
                plt.plot(data_HierDNN2[:, 1], data_HierDNN2[:, 3], linestyle_list[2+i], color=colormap[2+i], marker=marker_list[2+i], markerfacecolor='None', label=f'Residual learning', alpha=0.2)
                # plt.plot(data_HierDNN[:, 1], data_HierDNN[:, 3], linestyle_list[i], color=colormap[i], marker=marker_list[i], markerfacecolor='None', label=f'Proposed method', alpha=0.3)
            else:
                plt.plot(data_HierDNN2[:, 1], data_HierDNN2[:, 3], linestyle_list[2+i], color=colormap[2+i], marker=marker_list[2+i], markerfacecolor='None', alpha=0.2)
                # plt.plot(data_HierDNN[:, 1], data_HierDNN[:, 3], linestyle_list[i], color=colormap[i], marker=marker_list[i], markerfacecolor='None', alpha=0.3)
        
        for jj in range(num_iter):
            data_HierDNN = np.loadtxt(f'./figures/HierDNN/{date_list_HierDNN[jj]}/accuracy.csv', delimiter=',')
            if jj == 0:
                plt.plot(data_HierDNN[:, 1], data_HierDNN[:, 3], linestyle_list[i], color=colormap[i], marker=marker_list[i], markerfacecolor='None', label=f'Proposed method', alpha=0.2)
            else:
                plt.plot(data_HierDNN[:, 1], data_HierDNN[:, 3], linestyle_list[i], color=colormap[i], marker=marker_list[i], markerfacecolor='None', alpha=0.2)



    plt.grid()
    plt.xlabel('Number of parameters')
    plt.ylabel('RMSE')
    plt.xscale('log')
    # plt.yscale('log')
    # plt.legend()
    plt.savefig(FIGURE_FOLDER + 'fig_comparison_RMSE.png')
    plt.savefig(FIGURE_FOLDER + 'fig_comparison_RMSE.pdf')
    plt.show()


if __name__ == '__main__':
    # main_average()
    # main_LTI()
    main_HierLTIandHierDNN()
    # main6()
    # main5()