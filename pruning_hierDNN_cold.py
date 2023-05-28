#!/usr/bin/env python
# coding: utf-8

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import matplotlib.pyplot as plt

from MyNets.hierarchical_dnn import HierarchicalNeuralNetwork
from MyNets.dnn import DeepNeuralNetwork

from MyUtilities.try_gpu import try_gpu
from MyUtilities.measure_elapsed_time import tic, toc
from MyUtilities.rmse import rmse
from MyUtilities.fit import fit
from MyUtilities.load_csv_as_tensor import load_csv_as_tensor
from MyUtilities.standardize import standardize

def main(num_layer, hidden_size, date):
    # 初めから小さい構造で学習した階層的DNNを比較

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # date = f'{date}-{num_layer}-{hidden_size}'

    model_struc = f'{num_layer}-{hidden_size}'
    # model_struc = f'{num_layer}-{hidden_size}-cold'
    # model_struc = f'{num_layer}-{hidden_size}-california'

    FIGURE_FOLDER = f'./figures/HierDNN/residuals/sinc/{date}-{model_struc}/'

    outfile_path = FIGURE_FOLDER + 'accuracy.csv'   # 適合率の出力ファイルパス

    if not os.path.exists(FIGURE_FOLDER):
        # ディレクトリが存在しない場合、ディレクトリを作成する
        os.makedirs(FIGURE_FOLDER)
        os.makedirs(f'{FIGURE_FOLDER}pruned')

    # 階層的DNN読み込み
    n_u = 1
    INPUT_SIZE = n_u
    OUTPUT_SIZE = 1
    NUM_LAYER = num_layer
    HIDDEN_SIZE = hidden_size

    models = []
    for i in range(2, NUM_LAYER+1):
        model_path = f'./TrainedModels/HierDNN_residual_{date}-{i}-{model_struc}.pth'    # 学習済みの階層的DNNのパス
        # model_path = f'./TrainedModels/colds/HierDNN_cold_{date}-{i}-{model_struc}.pth'    # 学習済みの階層的DNNのパス
        models.append(HierarchicalNeuralNetwork(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, output_size=OUTPUT_SIZE, num_layer=NUM_LAYER, nonlinearity='relu'))
        models[i-2].load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        models[i-2] = try_gpu(models[i-2])

    # パラメータ数計算
    params_sum = 0
    for p in models[-1].parameters():
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

    # 対象システム定義
    N = 1000
    u = torch.linspace(start=-10, end=10, steps=N, device=device)
    u = u.reshape(-1, 1)
    y = torch.sin(u) / u

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

    # # u = data_train[:, 0:n_u]
    # # y = data_train[:, -1].reshape(-1,1)
    # u = data_val[:, 0:n_u]
    # y = data_val[:, -1].reshape(-1,1)
    
    amount_log = []
    params_log = []
    fit_log = []
    rmse_log = []

    for i in range(2, NUM_LAYER+1):
        yhat, _intermediate_y_hat = models[i-2](u, num_used_layers=i)   # i層使用したモデルの出力

        # モデル出力をプロット
        plt.figure()
        plt.plot(u.detach().cpu().numpy(), y.detach().cpu().numpy(), '-k', label='True value')
        plt.plot(u.detach().cpu().numpy(), yhat.detach().cpu().numpy(), '-.r', label='Model output')
        plt.grid()
        plt.xlabel('u')
        plt.ylabel('y')
        plt.legend()
        plt.savefig(f'{FIGURE_FOLDER}pruned/fig_validation_{i}.png')

        # 精度を計算
        fitrate = fit(yhat, y)
        rmse_hat = rmse(yhat, y).item()
        print(f'pruned: {amounts[i-2]*100} %, number of parameters: {params[i-2]},  fit: {fitrate}, rmse: {rmse_hat}')

        amount_log.append(amounts[i-2])
        params_log.append(params[i-2])
        fit_log.append(fitrate)
        rmse_log.append(rmse_hat)

    # ログ保存
    result = np.array([amount_log, params_log, fit_log, rmse_log]).transpose()
    np.savetxt(outfile_path, result, delimiter=',')

    # plt.show()



if __name__ == '__main__':
    # main(num_layer=5, hidden_size=64, date='2023-01-09')

    for trial in range(0, 10):
        for i in range(2, 6):
            print(i)
            date = '2023-01-18'+'-'+str(trial)
            main(num_layer=5, hidden_size=64, date=date)

        # for i in range(2, 18):
        #     print(i)
        #     date = '2023-01-18'+'-'+str(trial)
        #     main(num_layer=17, hidden_size=23, date=date)
