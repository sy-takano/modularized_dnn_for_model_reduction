#!/usr/bin/env python
# coding: utf-8

import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import matplotlib.pyplot as plt

from MyNets.lti import HierarchicalLTI, HierarchicalLTI2, HierarchicalLTIwithStateTransformer, HierarchicalLTIwithKalmanFilter
from MyUtilities.my_dataloader import InOutDataSilverbox
from MyUtilities.try_gpu import try_gpu
from MyUtilities.measure_elapsed_time import tic, toc
from MyUtilities.rmse import rmse
from MyUtilities.fit import fit


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


# 階層的な線形状態空間モデルの枝刈り（低次元化）を行う


# データのパスを指定
DATA_PATH= './data/csv/2022-10-05_oscil100/inoutdata_multimodes.csv'

# モデル読み込み
date = '2022-11-23-4_4_oscil100_3000_losstime10'
FIGURE_FOLDER = './figures/HierLTI/' + date + '/'
model_path = './TrainedModels/HierLTI_' + date + '.pth'

outfile_path = FIGURE_FOLDER + 'accuracy.csv'

NUM_LAYER = 4

# model = HierarchicalLTIwithKalmanFilter(num_layer=NUM_LAYER, u_size=1, y_size=1, order_of_a_LTImodel=2, send_gpu_state=False)
model = HierarchicalLTI2(num_layer=NUM_LAYER, u_size=1, y_size=1, order_of_a_LTImodel=2)

model = try_gpu(model)
model.A, model.B, model.C, model.D = model.make_ABCDmatrices(NUM_LAYER)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.A, model.B, model.C, model.D = model.make_ABCDmatrices(NUM_LAYER)


# パラメータ数計算
params = []
for i in range(NUM_LAYER):
    params_i = 0
    for p in model.lti_list[i].parameters():
        if p.requires_grad:
            params_i += p.numel()

    if i==0:
        params.append(params_i)
    else:
        params.append(params[-1] + params_i)

params = np.array(params)
amounts = 1 - params/params[-1]
print(params)


# データ読み込み
data = InOutDataSilverbox(DATA_PATH, send_gpu=False)
# data.input = data.input[9000:]
# data.output = data.output[9000:]

data.time = torch.arange(0, len(data.input), 1)


# 枝刈りして出力をプロット
bound = 5000   # はじめ9000データは学習データ
amount_log = []
params_log = []
fit_log = []
rmse_log = []
for i in range(NUM_LAYER):
    # 学習データに対するモデル出力を計算
    yhat = model(data.input[:bound], data.output[:bound], num_used_layers=i+1)

    fitrate = fit(yhat.detach().cpu(), data.output[:bound].detach().cpu())

    plt.figure()
    plt.plot(data.time[:bound].detach().cpu().numpy(), data.output[:bound].detach().cpu().numpy(), '-k', label='True value')
    plt.plot(data.time[:bound].detach().cpu().numpy(), yhat.detach().cpu().numpy(), '-.r', label=f'Model output ({fitrate:.1f} %)')
    plt.grid()
    plt.xlabel('Time k')
    plt.ylabel('Output y')
    plt.legend()
    plt.xlim(0, 1000)
    plt.savefig(f'{FIGURE_FOLDER}pruned/fig_train_zoom_{i}.png')


    # 検証データに対するモデル出力を計算
    yhat = model(data.input[bound:], data.output[bound:], num_used_layers=i+1)

    fitrate = fit(yhat.detach().cpu(), data.output[bound:].detach().cpu())
    rmse_hat = rmse(yhat.detach().cpu(), data.output[bound:].detach().cpu()).item()

    plt.figure()
    plt.plot(data.time[bound:].detach().cpu().numpy(), data.output[bound:].detach().cpu().numpy(), '-k', label='True value')
    plt.plot(data.time[bound:].detach().cpu().numpy(), yhat.detach().cpu().numpy(), '-.r', label=f'Model output ({fitrate:.1f} %)')
    plt.grid()
    plt.xlabel('Time k')
    plt.ylabel('Output y')
    plt.legend()
    plt.xlim(bound, bound+1000)
    plt.savefig(f'{FIGURE_FOLDER}pruned/fig_validation_zoom_{i}.png')


    plt.figure()
    plt.plot(data.time[bound:].detach().cpu().numpy(), yhat.detach().cpu().numpy() - data.output[bound:].detach().cpu().numpy(), label=f'Error (RMSE: {rmse_hat:.4f})')
    plt.grid()
    plt.xlabel('Time k')
    plt.ylabel('Error \hat{y} - y')
    plt.legend()
    plt.xlim(bound, bound+1000)
    plt.savefig(f'{FIGURE_FOLDER}pruned/fig_validation_error_zoom_{i}.png')

    # 適合率を計算
    print(f'pruned: {amounts[i]*100} %, number of parameters: {params[i]},  fit: {fitrate}, rmse: {rmse_hat}')

    amount_log.append(amounts[i])
    params_log.append(params[i])
    fit_log.append(fitrate)
    rmse_log.append(rmse_hat)


# ログを保存
result = np.array([amount_log, params_log, fit_log, rmse_log]).transpose()
np.savetxt(outfile_path, result, delimiter=',')

plt.show()
