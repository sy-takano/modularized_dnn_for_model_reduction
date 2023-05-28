#!/usr/bin/env python
# coding: utf-8

import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import matplotlib.pyplot as plt

from MyNets.dnn import DeepNeuralNetwork

from MyUtilities.try_gpu import try_gpu
from MyUtilities.measure_elapsed_time import tic, toc
from MyUtilities.rmse import rmse
from MyUtilities.fit import fit


# 学習済みのFNNの枝刈りを行う


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

date = '2022-04-25-17-23'
FIGURE_FOLDER = './figures/DNN/' + date + '/'
model_path = './TrainedModels/DNN_sinc_' + date + '.pth'
outfile_path = FIGURE_FOLDER + 'accuracy.csv'

# モデル読み込み
NUM_LAYER = 17
HIDDEN_SIZE = 23
model = DeepNeuralNetwork(input_size=1, hidden_size=HIDDEN_SIZE, output_size=1, num_layer=NUM_LAYER, nonlinearity='relu')
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model = try_gpu(model)

# 総パラメータ数表示
params = 0
for p in model.parameters():
    if p.requires_grad:
        params += p.numel()
print(f'number of parameters: {params}')

# 対象システム定義
N = 1000
u = torch.linspace(start=-10, end=10, steps=N, device=device)
u = u.reshape(-1, 1)
y = torch.sin(u) / u


yhat = model(u)
print(rmse(yhat, y).item())


amount_log = []
params_log = []
fit_log = []
rmse_log = []

amounts = np.concatenate([np.arange(0.00023, 0.1, 0.005), np.arange(0.1, 1.0+0.1, 0.1)], 0)
# amounts = np.arange(0.0, 1.0+0.05, 0.05)
amounts = 1.0 - amounts # 枝刈りするパラメータ数の割合
for amount in amounts:
    # モデル読み込み
    model = DeepNeuralNetwork(input_size=1, hidden_size=HIDDEN_SIZE, output_size=1, num_layer=NUM_LAYER, nonlinearity='relu')
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model = try_gpu(model)

    # 枝刈り対象のパラメータ
    parameters_to_prune = (
        (model.fc_list[0], 'weight'),
        (model.fc_list[0], 'bias'),
        (model.fc_list[1], 'weight'),
        (model.fc_list[1], 'bias'),
        (model.fc_list[2], 'weight'),
        (model.fc_list[2], 'bias'),
        (model.fc_list[3], 'weight'),
        (model.fc_list[3], 'bias')
    )
    # parameters_to_prune = (
    #     (model.fc_list[0], 'weight'),
    #     (model.fc_list[0], 'bias'),
    #     (model.fc_list[1], 'weight'),
    #     (model.fc_list[1], 'bias'),
    #     (model.fc_list[2], 'weight'),
    #     (model.fc_list[2], 'bias'),
    #     (model.fc_list[3], 'weight'),
    #     (model.fc_list[3], 'bias'),
    #     (model.fc_list[4], 'weight'),
    #     (model.fc_list[4], 'bias'),
    #     (model.fc_list[5], 'weight'),
    #     (model.fc_list[5], 'bias'),
    #     (model.fc_list[6], 'weight'),
    #     (model.fc_list[6], 'bias'),
    #     (model.fc_list[7], 'weight'),
    #     (model.fc_list[7], 'bias'),
    #     (model.fc_list[8], 'weight'),
    #     (model.fc_list[8], 'bias'),
    #     (model.fc_list[9], 'weight'),
    #     (model.fc_list[9], 'bias'),
    #     (model.fc_list[10], 'weight'),
    #     (model.fc_list[10], 'bias'),
    #     (model.fc_list[11], 'weight'),
    #     (model.fc_list[11], 'bias'),
    #     (model.fc_list[12], 'weight'),
    #     (model.fc_list[12], 'bias'),
    #     (model.fc_list[13], 'weight'),
    #     (model.fc_list[13], 'bias'),
    #     (model.fc_list[14], 'weight'),
    #     (model.fc_list[14], 'bias'),
    #     (model.fc_list[15], 'weight'),
    #     (model.fc_list[15], 'bias'),
    # )

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount
    )   # 絶対値の小さいパラメータから削除

    yhat = model(u)
    fitrate = fit(yhat, y)
    rmse_hat = rmse(yhat, y).item()
    print(f'pruned: {amount*100} %, number of parameters: {params*(1-amount)},  fit: {fitrate}, rmse: {rmse_hat}')

    amount_log.append(amount)
    params_log.append(params*(1-amount))
    fit_log.append(fitrate)
    rmse_log.append(rmse_hat)

    # 枝刈り後のモデルをプロット
    plt.figure()
    plt.plot(u.detach().cpu().numpy(), y.detach().cpu().numpy())
    plt.plot(u.detach().cpu().numpy(), yhat.detach().cpu().numpy())
    plt.grid()
    plt.xlabel('u')
    plt.ylabel('y')
    # plt.savefig(f'{FIGURE_FOLDER}pruned/fig_validation_{amount*100:.0f}.png')
    plt.savefig(FIGURE_FOLDER + 'pruned/fig_validation_' + str(amount) + '.png')

# ログを保存
result = np.array([amount_log, params_log, fit_log, rmse_log]).transpose()
np.savetxt(outfile_path, result, delimiter=',')

# plt.show()
