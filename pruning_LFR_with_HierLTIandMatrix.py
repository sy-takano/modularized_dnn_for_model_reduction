#!/usr/bin/env python
# coding: utf-8

import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import matplotlib.pyplot as plt

from MyNets.rnn_lfr import HierarchicalLTILFR
from MyNets.dnn import DeepNeuralNetwork

from MyUtilities.my_dataloader import InOutData, InOutDataSilverbox
from MyUtilities.try_gpu import try_gpu
from MyUtilities.measure_elapsed_time import tic, toc
from MyUtilities.rmse import rmse
from MyUtilities.fit import fit


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


# 階層的LTIと行列フィードバックで構成されるLFRモデルの枝刈りを行う

# データのパスを指定
DATA_PATH = 'data/csv/2022-10-05_oscil100/inoutdata_multimodes.csv'

# モデル読み込み
date = '2022-10-27'
FIGURE_FOLDER = './figures/LFR_with_HierLTIandMatrix/' + date + '/'
model_path = './TrainedModels/LFR_with_HierLTIandMatrix_' + date + '.pth'

outfile_path = FIGURE_FOLDER + 'accuracy.csv'

NUM_LAYER = 4
HIDDEN_SIZE = 64
X_SIZE = 2
INPUT_SIZE = 1
OUTPUT_SIZE = 1

model = HierarchicalLTILFR(num_layer_NN=5, hidden_size=64, num_layer_LTI=NUM_LAYER, u_size=1, y_size=1, w_size=1, z_size=1)
model.net = nn.Linear(1, 1, bias=False)

model = try_gpu(model)
model.A, model.B, model.C, model.D = model.make_ABCDmatrices(NUM_LAYER)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.A, model.B, model.C, model.D = model.make_ABCDmatrices(NUM_LAYER)



# パラメータ数計算
params_net = 0
for p in model.net.parameters():
    if p.requires_grad:
        params_net += p.numel()
print(params_net)   # 行列部分のパラメータ数（=1）

u_size = INPUT_SIZE
y_size = OUTPUT_SIZE
w_size = 1
z_size = 1

params_LTI = X_SIZE*X_SIZE + X_SIZE*(u_size + w_size) + y_size*X_SIZE + z_size*X_SIZE   # LTIモデル部分のパラメータ数
params_sum = params_net + params_LTI
print(params_LTI)
print(params_sum)


# データ読み込み
data = InOutDataSilverbox(DATA_PATH)
data.time = torch.arange(0, len(data.input), 1)


# 学習されたNNを学習データの範囲でプロット
bound = 9000   # はじめ9000データは学習データ
yhat = model(data.input[:bound], num_used_layers=NUM_LAYER)

# N = 10000
# z = torch.linspace(start=min(model.z_record).item(), end=max(model.z_record).item(), steps=N, device=device).reshape(-1, 1)
# w = model.net(z)

# w_max = max(w).detach().cpu().numpy()
# w_min = min(w).detach().cpu().numpy()

# plt.figure()
# plt.plot(z.detach().cpu().numpy(), w.detach().cpu().numpy(), '-k', label='w')
# plt.grid()
# plt.xlabel('z')
# plt.ylabel('w')
# plt.ylim(w_min - (w_max - w_min)*0.1, w_max + (w_max - w_min)*0.1)
# plt.legend()
# plt.savefig(f'{FIGURE_FOLDER}pruned/fig_function.png')


# 枝刈りして出力をプロット
fit_log = []
rmse_log = []
for i in range(NUM_LAYER):
    # 学習データに対するモデル出力を計算
    yhat = model(data.input[:bound], num_used_layers=i+1)

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
    yhat = model(data.input[bound:], num_used_layers=i+1)

    fitrate = fit(yhat.detach().cpu(), data.output[bound:].detach().cpu())

    plt.figure()
    plt.plot(data.time[bound:].detach().cpu().numpy(), data.output[bound:].detach().cpu().numpy(), '-k', label='True value')
    plt.plot(data.time[bound:].detach().cpu().numpy(), yhat.detach().cpu().numpy(), '-.r', label=f'Model output ({fitrate:.1f} %)')
    plt.grid()
    plt.xlabel('Time k')
    plt.ylabel('Output y')
    plt.legend()
    plt.xlim(bound, bound+1000)
    plt.savefig(f'{FIGURE_FOLDER}pruned/fig_validation_zoom_{i}.png')


    # 適合率を計算
    fitrate = fit(yhat.detach().cpu(), data.output[bound:].detach().cpu())
    rmse_hat = rmse(yhat.detach().cpu(), data.output[bound:].detach().cpu()).item()
    # print(f'pruned: {amounts[i]*100} %, number of parameters: {params[i]},  fit: {fitrate}, rmse: {rmse_hat}')
    print(f'Number of LTI modules: {i}, fit: {fitrate}, rmse: {rmse_hat}')

    fit_log.append(fitrate)
    rmse_log.append(rmse_hat)


# ログを保存
# result = np.array([amount_log, params_log, fit_log, rmse_log]).transpose()
result = np.array([fit_log, rmse_log]).transpose()
np.savetxt(outfile_path, result, delimiter=',')


# LTIモデルのパラメータを表示
# print(f'A: {model.A.weight}')
# print(f'Bu: {model.Bu.weight}')
# print(f'Bw: {model.Bw.weight}')
# print(f'Cy: {model.Cy.weight}')
# print(f'Cz: {model.Cz.weight}')
# print(f'Dyu: {model.Dyu.weight}')
# print(f'Dyw: {model.Dyw.weight}')
# print(f'Dzu: {model.Dzu.weight}')
# print(f'Dzw: {model.Dzw.weight}')

plt.show()
