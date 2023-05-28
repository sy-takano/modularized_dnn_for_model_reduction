#!/usr/bin/env python
# coding: utf-8

import imp
import os
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from copy import deepcopy
from datetime import date
from MyUtilities.fit import fit
from MyUtilities.load_csv_as_tensor import load_csv_as_tensor
from MyUtilities.standardize import standardize

from MyUtilities.try_gpu import try_gpu
from MyUtilities.measure_elapsed_time import tic, toc
from MyUtilities.rmse import rmse
from MyUtilities.get_grad import get_grad, plot_histogram

from MyNets.dnn import DeepNeuralNetwork


def static_system(u:torch.tensor):
    # 学習対象の静的システム
    # y = torch.sin(u) / u
    # y = u[:, 0] +2*u[:, 1]
    x1 = u[:, 0]
    x2 = u[:, 1]
    y = x1*torch.exp(-torch.pow(x1, 2)-torch.pow(x2,2))
    y = y.reshape(-1,1)
    # y = step(u)
    return y

def step(x):
  return 1.0 * (x >= 0.0)



def main(num_layer, hidden_size):
    # 静的関数を対象としてFNNの学習を行う

    NUM_LAYER = num_layer
    HIDDEN_SIZE = hidden_size
    n_u = 8

    AFFIX = f'-{NUM_LAYER}-{HIDDEN_SIZE}-california'

    FIGURE_FOLDER = './figures/DNN/' + date.today().strftime('%Y-%m-%d') + AFFIX + '/'

    MODEL_FOLDER = './TrainedModels/'
    model_name = 'DNN_california_' + date.today().strftime('%Y-%m-%d') + AFFIX + '.pth'
    model_path = MODEL_FOLDER + model_name

    if not os.path.exists(FIGURE_FOLDER):
        # ディレクトリが存在しない場合、ディレクトリを作成する
        os.makedirs(FIGURE_FOLDER)
        os.makedirs(f'{FIGURE_FOLDER}grad_hist')
        os.makedirs(f'{FIGURE_FOLDER}pruned')


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)


    model = DeepNeuralNetwork(input_size=n_u, hidden_size=HIDDEN_SIZE, output_size=1, num_layer=NUM_LAYER, nonlinearity='relu')
    model = try_gpu(model)


    # データ読み込み/生成
    DATAPATH = 'data/csv'
    DATANAME_TRAIN = 'cadata.csv'
    DATANAME_VAL = 'cadata.csv'
    PATH_TRAIN = DATAPATH + '/' + DATANAME_TRAIN
    PATH_VAL = DATAPATH + '/' + DATANAME_VAL
    data_train = try_gpu(standardize(load_csv_as_tensor(PATH_TRAIN)))
    data_val = try_gpu(standardize(load_csv_as_tensor(PATH_VAL)))

    N_all = len(data_train)
    np.random.seed(0)
    randomize_index = np.random.permutation(N_all)
    data_train = data_train[randomize_index]
    data_val = data_val[randomize_index]

    data_train = data_train[:15000]
    data_val = data_val[15000:]

    u = data_train[:, 0:n_u]
    y = data_train[:, -1].reshape(-1,1)
    u_val = data_val[:, 0:n_u]
    y_val = data_val[:, -1].reshape(-1,1)
    
    # N = 1000
    # rng = 2
    # u = 2*rng*(torch.rand(n_u*N, device=device) - 0.5).reshape(N, n_u)   # u ~ Uniform [-rng, rng)
    # u_val = 2*rng*(torch.rand(n_u*N, device=device) - 0.5).reshape(N, n_u)   # u ~ Uniform [-rng, rng)
    # # u_val = torch.linspace(start=-10, end=10, steps=N, device=device).reshape(-1, 1)   # u ~ Uniform [-10, 10)

    # # システム定義
    # y = static_system(u)
    # y_val = static_system(u_val)



    EPOCHS = 20000


    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.0001)


    loss_history = []
    grad_history = []
    epoch_history = []
    fit_val_history = []
    tic()   # 学習時間測定開始
    for epoch in range(EPOCHS):
        optimizer.zero_grad()

        yhat = model(u)
        loss = criterion(yhat, y)
        loss.backward()
        optimizer.step()

        grad = get_grad(model.parameters())

        if epoch % 100 == 0:
            print('epoch: ', epoch, ' loss (train): ', loss.item(), 'grad: ', grad.item())
        
        if epoch % 1000 == 0:
            # 勾配の分布をプロット
            plot_histogram(model.parameters())
            plt.grid()
            plt.savefig(f'{FIGURE_FOLDER}grad_hist/fig_hist_{epoch}.png')
            plt.close()

        yhat_val = model(u_val)

        loss_history.append(loss.item())
        grad_history.append(grad.item())
        epoch_history.append(epoch)

        fit_val_history.append(fit(yhat_val, y_val))


    toc()   # 学習時間測定終了

    # 学習済みモデルを保存
    model_for_save = deepcopy(model)
    torch.save(model_for_save.to('cpu').state_dict(), model_path)
    print('The trained model has been saved at ' + model_path)

    # 学習ログを保存
    log_train = np.array([epoch_history, loss_history, grad_history, fit_val_history]).transpose()
    np.savetxt(FIGURE_FOLDER+'log.csv', log_train, delimiter=',')
    print('The log has been saved at ' + FIGURE_FOLDER + 'log.csv')

    # 損失関数の変化をプロット
    plt.figure()
    plt.plot(loss_history)
    plt.yscale('log')
    plt.grid()
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.savefig(FIGURE_FOLDER + 'fig_loss.png')

    # 勾配の変化をプロット
    plt.figure()
    plt.plot(grad_history)
    plt.yscale('log')
    plt.grid()
    plt.xlabel('Epoch')
    plt.ylabel('grad')
    plt.savefig(FIGURE_FOLDER + 'fig_grad.png')


    # モデルを検証：真のシステムと比較
    # u = torch.linspace(start=-10, end=10, steps=N, device=device)
    # u = u.reshape(-1, 1)
    # y = static_system(u)
    yhat = model(u)
    plt.figure()
    plt.plot(u.detach().cpu().numpy(), y.detach().cpu().numpy(), '-k', label='True value')
    plt.plot(u.detach().cpu().numpy(), yhat.detach().cpu().numpy(), '-.r', label='Model output')
    plt.grid()
    plt.xlabel('u')
    plt.ylabel('y')
    plt.legend()
    plt.savefig(FIGURE_FOLDER + 'fig_validation.png')


    print(rmse(yhat, y).item())


    # パラメータ数を表示
    params = 0
    for p in model.parameters():
        if p.requires_grad:
            params += p.numel()
    print(params)    

    # plt.show()


if __name__ == '__main__':
    main(num_layer=6, hidden_size=23)

