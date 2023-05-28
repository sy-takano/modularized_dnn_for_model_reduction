#!/usr/bin/env python
# coding: utf-8

import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import matplotlib.pyplot as plt
from copy import deepcopy

from MyNets.hierarchical_dnn import HierarchicalNeuralNetwork
from MyNets.dnn import DeepNeuralNetwork
from MyNets.trainer import retrain_signal_regularization

from MyUtilities.try_gpu import try_gpu
from MyUtilities.measure_elapsed_time import tic, toc
from MyUtilities.rmse import rmse
from MyUtilities.fit import fit
from MyUtilities.load_csv_as_tensor import load_csv_as_tensor
from MyUtilities.standardize import standardize


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


# 学習済みの階層的DNNの縮約を行った後に再学習

def main(num_layer, hidden_size, date):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    date = f'{date}-{num_layer}-{hidden_size}-california'
    FIGURE_FOLDER = './figures/HierDNN/' + date + '/retrained/'
    model_path = './TrainedModels/HierDNN_california_' + date + '.pth'    # 学習済みの階層的DNNのパス

    outfile_path = FIGURE_FOLDER + 'accuracy.csv'   # 精度の出力ファイルパス

    # 階層的DNN読み込み
    INPUT_SIZE = 8
    OUTPUT_SIZE = 1
    NUM_LAYER = num_layer
    HIDDEN_SIZE = hidden_size
    model = HierarchicalNeuralNetwork(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, output_size=OUTPUT_SIZE, num_layer=NUM_LAYER, nonlinearity='relu')
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model = try_gpu(model)

    # パラメータ数計算
    params_sum = 0
    for p in model.parameters():
        if p.requires_grad:
            params_sum  += p.numel()
    print(params_sum)

    params = []
    params.append(INPUT_SIZE*OUTPUT_SIZE + OUTPUT_SIZE)
    params.append(params[-1] + INPUT_SIZE*HIDDEN_SIZE + HIDDEN_SIZE + HIDDEN_SIZE*OUTPUT_SIZE + OUTPUT_SIZE)
    for i in range(1, NUM_LAYER-2):
        params.append(params[-1] + HIDDEN_SIZE*HIDDEN_SIZE + HIDDEN_SIZE + HIDDEN_SIZE*OUTPUT_SIZE + OUTPUT_SIZE)   # i+1層の階層的NNのパラメータ数

    params = np.array(params)
    amounts = 1 - params/params_sum
    print(params)

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

    n_u = INPUT_SIZE
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

    amount_log = []
    params_log = []
    fit_retrained_log = []
    rmse_retrained_log = []
    fit_before_retrained_log = []
    rmse_before_retrained_log = []
    EPOCHS = 10000

    for i in range(NUM_LAYER-1):
        # 階層的DNN読み込み
        model = HierarchicalNeuralNetwork(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, output_size=OUTPUT_SIZE, num_layer=NUM_LAYER, nonlinearity='relu')
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model = try_gpu(model)

        yhat_val, _intermediate_y_hat = model(u_val, num_used_layers=i+2)   # i層使用したモデルの出力

        # モデル出力をプロット
        plt.figure()
        plt.plot(u_val.detach().cpu().numpy(), y_val.detach().cpu().numpy(), '-k', label='True value')
        plt.plot(u_val.detach().cpu().numpy(), yhat_val.detach().cpu().numpy(), '--r', label='Model output')
        plt.grid()
        plt.xlabel('u')
        plt.ylabel('y')
        plt.legend()
        plt.savefig(f'{FIGURE_FOLDER}fig_validation_before_retrained_{i}.png')

        # 精度を計算
        fitrate_before_retrained = fit(yhat_val, y_val)
        rmse_hat_before_retrained = rmse(yhat_val, y_val).item()
        print(f'pruned: {amounts[i]*100} %, number of parameters: {params[i]},  fit_before_retrained: {fitrate_before_retrained}, rmse_before_retrained: {rmse_hat_before_retrained}')

        tic()   # 学習時間測定開始
        loss_history, epoch_history, grad_history, fit_val_history = retrain_signal_regularization(model, u, y, num_layer=NUM_LAYER, path_histogram=f'{FIGURE_FOLDER}grad_hist', num_used_layer=i+2, u_val=u_val, y_val=y_val, epochs=EPOCHS, device=device)
        toc()   # 学習時間測定終了

        yhat_val, _intermediate_y_hat = model(u_val, num_used_layers=i+2)   # i層使用したモデルの出力

        # モデル出力をプロット
        plt.figure()
        plt.plot(u_val.detach().cpu().numpy(), y_val.detach().cpu().numpy(), '-k', label='True value')
        plt.plot(u_val.detach().cpu().numpy(), yhat_val.detach().cpu().numpy(), '--r', label='Model output')
        plt.grid()
        plt.xlabel('u')
        plt.ylabel('y')
        plt.legend()
        plt.savefig(f'{FIGURE_FOLDER}fig_validation_retrained_{i}.png')

        # 精度を計算
        fitrate_retrained = fit(yhat_val, y_val)
        rmse_hat_retrained = rmse(yhat_val, y_val).item()
        print(f'pruned: {amounts[i]*100} %, number of parameters: {params[i]},  fit: {fitrate_retrained}, rmse: {rmse_hat_retrained}')

        amount_log.append(amounts[i])
        params_log.append(params[i])
        fit_retrained_log.append(fitrate_retrained)
        rmse_retrained_log.append(rmse_hat_retrained)
        fit_before_retrained_log.append(fitrate_before_retrained)
        rmse_before_retrained_log.append(rmse_hat_before_retrained)

        # 学習済みモデルを保存
        model_for_save = deepcopy(model)
        model_path_retrained = f'{FIGURE_FOLDER}retrained_{i}.pth'
        torch.save(model_for_save.to('cpu').state_dict(), model_path_retrained)
        print('The trained model has been saved at ' + model_path_retrained)

        # 学習ログを保存
        log_train = np.array([epoch_history, loss_history, grad_history, fit_val_history]).transpose()
        log_path = f'{FIGURE_FOLDER}log_{i}.csv'
        np.savetxt(log_path, log_train, delimiter=',')
        print('The log has been saved at ' + log_path)
        
        # 損失関数の変化をプロット
        plt.figure()
        plt.plot(loss_history)
        plt.yscale('log')
        plt.grid()
        plt.xlabel('Epoch')
        plt.ylabel('loss')
        plt.legend()
        plt.savefig(f'{FIGURE_FOLDER}fig_loss_{i}.png')



    # ログ保存
    result = np.array([amount_log, params_log, fit_retrained_log, rmse_retrained_log, fit_before_retrained_log, rmse_before_retrained_log]).transpose()
    np.savetxt(outfile_path, result, delimiter=',')

    plt.show()



if __name__ == '__main__':
    main(num_layer=17, hidden_size=23, date='2022-09-21')
