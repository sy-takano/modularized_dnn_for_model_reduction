#!/usr/bin/env python
# coding: utf-8

import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import matplotlib.pyplot as plt

from MyNets.dnn import DeepNeuralNetwork
from MyNets.rnn_lfr import DNNLFR
from MyUtilities.my_dataloader import InOutData, InOutDataSilverbox

from MyUtilities.try_gpu import try_gpu
from MyUtilities.measure_elapsed_time import tic, toc
from MyUtilities.rmse import rmse
from MyUtilities.fit import fit

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


# 非線形LFRとFNNを組み合わせたモデルの枝刈りを行う

# データのパスを指定
DATA_PATH= 'data/csv/SNLS80mV.csv'

# モデル読み込み
date = '2022-04-20'
FIGURE_FOLDER = './figures/LFR_with_DNN/' + date + '/'
model_path = './TrainedModels/LFR_with_DNN_' + date + '.pth'
outfile_path = FIGURE_FOLDER + 'accuracy.csv'   # 精度ファイル書き出しパス

NUM_LAYER = 5
HIDDEN_SIZE = 64
X_SIZE = 2

model = DNNLFR(hidden_size=HIDDEN_SIZE, num_layer=NUM_LAYER, x_size=X_SIZE)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model = try_gpu(model)


# パラメータ数計算
params_net = 0
for p in model.net.parameters():
    if p.requires_grad:
        params_net += p.numel()
print(params_net)   # FNN部分のパラメータ数

u_size = 1
y_size = 1
w_size = 1
z_size = 1

params_LTI = X_SIZE*X_SIZE + X_SIZE*(u_size + w_size) + y_size*X_SIZE + z_size*X_SIZE   # LTIモデル部分のパラメータ数
params_sum = params_net + params_LTI
print(params_LTI)
print(params_sum)

INPUT_SIZE = 1
OUTPUT_SIZE = 1
params = []
params.append(INPUT_SIZE*OUTPUT_SIZE + OUTPUT_SIZE)
params.append(params[-1] + INPUT_SIZE*HIDDEN_SIZE + HIDDEN_SIZE + HIDDEN_SIZE*OUTPUT_SIZE + OUTPUT_SIZE)
for i in range(1, NUM_LAYER-2):
    params.append(params[-1] + HIDDEN_SIZE*HIDDEN_SIZE + HIDDEN_SIZE + HIDDEN_SIZE*OUTPUT_SIZE + OUTPUT_SIZE)   # i+1層の階層的NNのパラメータ数

params = np.array(params)
amounts = 1 - params/params_net
for i, amount in enumerate(amounts):
    if amount < 0:
        amounts[i] = 0

# データ読み込み
data = InOutDataSilverbox(DATA_PATH)

data.input = data.input[50000:]     # マルチサイン入力の部分のみ使う
data.output = data.output[50000:]

data.time = torch.arange(0, len(data.input), 1)

bound = 20000   # 学習データと検証データの境目


amount_log = []
params_log = []
fit_log = []
rmse_log = []


# amounts = np.arange(0.0, 1.0+0.05, 0.05)
# amounts = 1.0 - amounts # 枝刈りを行うパラメータ数の割合
for amount in amounts:
    params = params_LTI + params_net*(1-amount)

    model = DNNLFR(hidden_size=HIDDEN_SIZE, num_layer=NUM_LAYER, x_size=X_SIZE)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model = try_gpu(model)

    # 枝刈り対象のパラメータ FNNの部分のみ枝刈り
    parameters_to_prune = (
        (model.net.fc_list[0], 'weight'),
        (model.net.fc_list[0], 'bias'),
        (model.net.fc_list[1], 'weight'),
        (model.net.fc_list[1], 'bias'),
        (model.net.fc_list[2], 'weight'),
        (model.net.fc_list[2], 'bias'),
        (model.net.fc_list[3], 'weight'),
        (model.net.fc_list[3], 'bias')
    )

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount
    )   # 絶対値の小さいパラメータから削除

    
    # 学習データに対してモデル出力と適合率を計算
    yhat = model(data.input[:bound])
    fitrate = fit(yhat.detach().cpu(), data.output[:bound].detach().cpu())

    # 出力プロット
    plt.figure()
    plt.plot(data.time[:bound].detach().cpu().numpy(), data.output[:bound].detach().cpu().numpy(), '-k', label='True value')
    plt.plot(data.time[:bound].detach().cpu().numpy(), yhat.detach().cpu().numpy(), '-.r', label=f'Model output ({fitrate:.1f} %)')
    plt.grid()
    plt.xlabel('Time k')
    plt.ylabel('Output y')
    plt.legend()
    plt.xlim(0, 100)
    plt.savefig(f'{FIGURE_FOLDER}pruned/fig_train_zoom_{amount}.png')


    # 検証データに対してモデル出力と適合率を計算
    yhat = model(data.input[bound:bound+20000])

    fitrate = fit(yhat.detach().cpu(), data.output[bound:bound+20000].detach().cpu())
    rmse_hat = rmse(yhat.detach().cpu(), data.output[bound:bound+20000].detach().cpu()).item()

    amount_log.append(amount)
    params_log.append(params)
    fit_log.append(fitrate)
    rmse_log.append(rmse_hat)

    # 出力プロット
    plt.figure()
    plt.plot(data.time[bound:bound+20000].detach().cpu().numpy(), data.output[bound:bound+20000].detach().cpu().numpy(), '-k', label='True value')
    plt.plot(data.time[bound:bound+20000].detach().cpu().numpy(), yhat.detach().cpu().numpy(), '-.r', label=f'Model output ({fitrate:.1f} %)')
    plt.grid()
    plt.xlabel('Time k')
    plt.ylabel('Output y')
    plt.legend()
    plt.xlim(bound, bound+100)
    plt.savefig(f'{FIGURE_FOLDER}pruned/fig_validation_zoom_{amount}.png')

# ログ保存
result = np.array([amount_log, params_log, fit_log, rmse_log]).transpose()
np.savetxt(outfile_path, result, delimiter=',')

# plt.show()
