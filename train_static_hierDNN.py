#!/usr/bin/env python
# coding: utf-8

import os
import sys
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

from copy import deepcopy
from datetime import date

from MyUtilities.try_gpu import try_gpu
from MyUtilities.measure_elapsed_time import tic, toc
from MyUtilities.rmse import rmse
from MyUtilities.fit import fit
from MyUtilities.load_csv_as_tensor import load_csv_as_tensor
from MyUtilities.standardize import standardize

from MyNets.hierarchical_dnn import HierarchicalNeuralNetwork
from MyNets.trainer import train_signal_regularization

def static_system(u:torch.tensor):
    # 学習対象の静的システム
    y = torch.sin(u) / u

    # y = u[:, 0] +2*u[:, 1]

    # x1 = u[:, 0]
    # x2 = u[:, 1]
    # y = x1*torch.exp(-torch.pow(x1, 2)-torch.pow(x2,2))
    # y = y.reshape(-1,1)
    
    # y = step(u)
    return y

def step(x):
  return 1.0 * (x >= 0.0)

def main(num_layer, hidden_size, date_folder, gamma=None):
    # 静的関数を対象として階層的DNNの学習を行う

    NUM_LAYER = num_layer
    HIDDEN_SIZE = hidden_size
    n_u = 1
    # n_u = 8

    AFFIX = f'-{NUM_LAYER}-{HIDDEN_SIZE}'
    # AFFIX = f'-{NUM_LAYER}-{HIDDEN_SIZE}-cold'
    # AFFIX = f'-{NUM_LAYER}-{HIDDEN_SIZE}-cold-california'
    # AFFIX = f'-{NUM_LAYER}-{HIDDEN_SIZE}-california'

    # FIGURE_FOLDER = './figures/HierDNN/' + date_folder + AFFIX + '/'
    FIGURE_FOLDER = './figures/HierDNN/proposed/' + date_folder + AFFIX + '/'
    # FIGURE_FOLDER = './figures/HierDNN/cold/california/' + date_folder + AFFIX + '/'

    MODEL_FOLDER = './TrainedModels/'
    # MODEL_FOLDER = './TrainedModels/colds/'
    model_name = 'HierDNN_sinc_' + date_folder + AFFIX + '.pth'
    # model_name = 'HierDNN_cold_' + date_folder + AFFIX + '.pth'
    # model_name = 'HierDNN_california_' + date_folder + AFFIX + '.pth'
    model_path = MODEL_FOLDER + model_name


    if not os.path.exists(FIGURE_FOLDER):
        # ディレクトリが存在しない場合、ディレクトリを作成する
        os.makedirs(FIGURE_FOLDER)
        os.makedirs(f'{FIGURE_FOLDER}grad_hist')
        os.makedirs(f'{FIGURE_FOLDER}pruned')


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')


    model = HierarchicalNeuralNetwork(input_size=n_u, hidden_size=HIDDEN_SIZE, output_size=1, num_layer=NUM_LAYER, nonlinearity='relu')
    model = try_gpu(model)


    # # データ読み込み/生成
    # DATAPATH = 'data/csv'
    # DATANAME_TRAIN = 'cadata.csv'
    # DATANAME_VAL = 'cadata.csv'
    # PATH_TRAIN = DATAPATH + '/' + DATANAME_TRAIN
    # PATH_VAL = DATAPATH + '/' + DATANAME_VAL
    # data_train = try_gpu(standardize(load_csv_as_tensor(PATH_TRAIN)))
    # data_val = try_gpu(standardize(load_csv_as_tensor(PATH_VAL)))

    # N_all = len(data_train)
    # # np.random.seed(0)
    # randomize_index = np.random.permutation(N_all)
    # data_train = data_train[randomize_index]
    # data_val = data_val[randomize_index]

    # data_train = data_train[:15000]
    # data_val = data_val[15000:]

    # u = data_train[:, 0:n_u]
    # y = data_train[:, -1].reshape(-1,1)
    # u_val = data_val[:, 0:n_u]
    # y_val = data_val[:, -1].reshape(-1,1)
    
    N = 1000
    rng = 10
    u = 2*rng*(torch.rand(n_u*N, device=device) - 0.5).reshape(N, n_u)   # u ~ Uniform [-rng, rng)
    # u_val = 2*rng*(torch.rand(n_u*N, device=device) - 0.5).reshape(N, n_u)   # u ~ Uniform [-rng, rng)
    u_val = torch.linspace(start=-10, end=10, steps=N, device=device).reshape(-1, 1)   # u ~ Uniform [-10, 10)

    # システム定義
    y = static_system(u)
    y_val = static_system(u_val)



    EPOCHS = 20000

    if gamma == None:
        gamma = [1. for i in range(NUM_LAYER - 1)]

    # gamma = [0. for i in range(NUM_LAYER - 1)]
    # gamma[-1] = 1.
    # # gamma = [1. for i in range(NUM_LAYER)]


    tic()   # 学習時間測定開始
    loss_history, loss_history_termwise, epoch_history, grad_history, fit_val_history = train_signal_regularization(model, u, y, num_layer=NUM_LAYER, regularization_parameter=gamma, path_histogram=f'{FIGURE_FOLDER}grad_hist', u_val=u_val, y_val=y_val, epochs=EPOCHS, device=device)
    toc()   # 学習時間測定終了


    # 学習済みモデルを保存
    model_for_save = deepcopy(model)
    torch.save(model_for_save.to('cpu').state_dict(), model_path)
    print('The trained model has been saved at ' + model_path)

    # 学習ログを保存
    log_train = np.array([epoch_history, loss_history, grad_history, fit_val_history]).transpose()
    np.savetxt(FIGURE_FOLDER+'log.csv', log_train, delimiter=',')
    print('The log has been saved at ' + FIGURE_FOLDER + 'log.csv')

    # 層ごとの損失関数を保存
    np.savetxt(FIGURE_FOLDER+'loss_termwise.csv', loss_history_termwise, delimiter=',')


    # 損失関数の変化をプロット
    plt.figure()
    plt.plot(loss_history)
    plt.yscale('log')
    plt.grid()
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(FIGURE_FOLDER + 'fig_loss.png')

    # 層ごとの損失関数の変化をプロット
    plt.figure()
    for i in range(NUM_LAYER-1):
        plt.plot(loss_history_termwise[i], linestyle='dashdot', label=f'term number = {i+1}')
    plt.yscale('log')
    plt.grid()
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(FIGURE_FOLDER + 'fig_loss_termwise.png')

    # 層ごとの損失関数をγで重みづけしてプロット
    plt.figure()
    for i in range(NUM_LAYER-1):
        plt.plot([gamma[i]*jj for jj in loss_history_termwise[i]], linestyle='dashdot', label=f'term number = {i+1}')
    plt.yscale('log')
    plt.grid()
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(FIGURE_FOLDER + 'fig_weightedloss_termwise.png')

    # 勾配の変化をプロット
    plt.figure()
    plt.plot(grad_history)
    plt.yscale('log')
    plt.grid()
    plt.xlabel('Epoch')
    plt.ylabel('grad')
    plt.savefig(FIGURE_FOLDER + 'fig_grad.png')
    
    yhat_val, intermediate_y_hat_val = model(u_val) # yhat: 全層使った出力，intermediate_y_hat: サブ出力

    print(rmse(yhat_val, y_val).item())
    print(fit(yhat_val, y_val))

    # 全層使った出力を真の関数と比較
    plt.figure()
    plt.plot(u_val.detach().cpu().numpy(), y_val.detach().cpu().numpy(), '-k', label='True value')
    plt.plot(u_val.detach().cpu().numpy(), yhat_val.detach().cpu().numpy(), '-.r', label='Model output')
    plt.grid()
    plt.xlabel('u')
    plt.ylabel('y')
    plt.legend()
    plt.savefig(FIGURE_FOLDER + 'fig_validation.png')

    # サブ出力を真の関数と比較
    plt.figure()
    plt.plot(u_val.detach().cpu().numpy(), yhat_val.detach().cpu().numpy(), label='y')
    for i in range(NUM_LAYER-1):
        plt.plot(u_val.detach().cpu().numpy(), intermediate_y_hat_val[i].detach().cpu().numpy(), label=f'o_{i+1}')
    plt.plot(u_val.detach().cpu().numpy(), 0.166*torch.ones_like(u_val).detach().cpu().numpy(), '-.k', label='BLA')
    plt.grid()
    plt.legend()
    plt.xlabel('u')
    plt.ylabel('y')
    plt.savefig(FIGURE_FOLDER + 'fig_validation_intermediate.png')


    # パラメータ数を表示
    params = 0
    for p in model.parameters():
        if p.requires_grad:
            params += p.numel()
    print(params)    

    # plt.show()

if __name__ == '__main__':
    # for trial in range(10):
    #     for i in range(5 - 1):
    #         gamma = [0. for i in range(5 - 1)]
    #         gamma[-1-i] = 1.
    #         print(gamma)
    #         # main(num_layer=5, hidden_size=64, date_folder='2023-01-24'+'-'+str(trial))
    #         main(num_layer=5, hidden_size=64, date_folder=date.today().strftime('%Y-%m-%d')+'-'+str(trial)+'-'+str(5-i), gamma=gamma)

    #     for i in range(17 - 1):
    #         gamma = [0. for i in range(17 - 1)]
    #         gamma[-1-i] = 1.
    #         print(gamma)
    #     #     main(num_layer=17, hidden_size=23, date_folder='2023-01-24'+'-'+str(trial))
    #         main(num_layer=17, hidden_size=23, date_folder=date.today().strftime('%Y-%m-%d')+'-'+str(trial)+'-'+str(17-i), gamma=gamma)
    
    for trial in range(10):
        main(num_layer=5, hidden_size=64, date_folder=date.today().strftime('%Y-%m-%d')+'-'+str(trial), gamma=[1, 1, 1, 0])
    # main(num_layer=5, hidden_size=64, date_folder=date.today().strftime('%Y-%m-%d'), gamma=[0.2, 0.4, 0.2, 0.2])

