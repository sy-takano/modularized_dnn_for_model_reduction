#!/usr/bin/env python
# coding: utf-8

import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import matplotlib.pyplot as plt

from MyNets.rnn_lfr import HierarichicalRNNLFR
from MyNets.dnn import DeepNeuralNetwork

from MyUtilities.my_dataloader import InOutData, InOutDataSilverbox
from MyUtilities.try_gpu import try_gpu
from MyUtilities.measure_elapsed_time import tic, toc
from MyUtilities.rmse import rmse
from MyUtilities.fit import fit


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


# 非線形LFRと階層的DNNを組み合わせたモデルの枝刈りを行う

# データのパスを指定
DATA_PATH= 'data/csv/SNLS80mV.csv'

# モデル読み込み
date = '2022-10-11'
FIGURE_FOLDER = './figures/LFR_with_hierDNN/' + date + '/'
model_path = './TrainedModels/LFR_with_hierDNN_' + date + '.pth'

outfile_path = FIGURE_FOLDER + 'accuracy.csv'

NUM_LAYER = 5
HIDDEN_SIZE = 64
X_SIZE = 2
INPUT_SIZE = 1
OUTPUT_SIZE = 1

model = HierarichicalRNNLFR(hidden_size=HIDDEN_SIZE, num_layer=NUM_LAYER, x_size=X_SIZE)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model = try_gpu(model)


# パラメータ数計算
params_net = 0
for p in model.net.parameters():
    if p.requires_grad:
        params_net += p.numel()
print(params_net)   # NN部分のパラメータ数

u_size = INPUT_SIZE
y_size = OUTPUT_SIZE
w_size = 1
z_size = 1

params_LTI = X_SIZE*X_SIZE + X_SIZE*(u_size + w_size) + y_size*X_SIZE + z_size*X_SIZE   # LTIモデル部分のパラメータ数
params_sum = params_net + params_LTI
print(params_LTI)
print(params_sum)

# 縮約後の階層的DNNのパラメータ数を計算
params = []
params.append(INPUT_SIZE*OUTPUT_SIZE + OUTPUT_SIZE)
params.append(params[-1] + INPUT_SIZE*HIDDEN_SIZE + HIDDEN_SIZE + HIDDEN_SIZE*OUTPUT_SIZE + OUTPUT_SIZE)
for i in range(2, NUM_LAYER-1):
    params.append(params[-1] + HIDDEN_SIZE*HIDDEN_SIZE + HIDDEN_SIZE + HIDDEN_SIZE*OUTPUT_SIZE + OUTPUT_SIZE)
params = np.array(params) + params_LTI
amounts = 1 - params/params_sum
print(params)


# データ読み込み
data = InOutDataSilverbox(DATA_PATH)
data.input = data.input[50000:]     # マルチサイン入力の部分のみ使う
data.output = data.output[50000:]

data.time = torch.arange(0, len(data.input), 1)


# 学習されたNNを学習データの範囲でプロット
bound = 20000   # はじめ20000データは学習データ
yhat = model(data.input[:bound], num_used_layers=4)

N = 10000
z = torch.linspace(start=min(model.z_record).item(), end=max(model.z_record).item(), steps=N, device=device).reshape(-1, 1)
w, intermediate_w = model.net(z, num_used_layers=NUM_LAYER)

w_max = max(w).detach().cpu().numpy()
w_min = min(w).detach().cpu().numpy()

plt.figure()
plt.plot(z.detach().cpu().numpy(), w.detach().cpu().numpy(), '-.r', label='w')
for i in range(NUM_LAYER-1):
    plt.plot(z.detach().cpu().numpy(), intermediate_w[i].detach().cpu().numpy(), label=f'w{i+1}')
plt.grid()
plt.xlabel('z')
plt.ylabel('w')
plt.ylim(w_min - (w_max - w_min)*0.1, w_max + (w_max - w_min)*0.1)
plt.legend()
plt.savefig(f'{FIGURE_FOLDER}pruned/fig_function.png')


# 枝刈りして出力をプロット
amount_log = []
params_log = []
fit_log = []
rmse_log = []
for i in range(NUM_LAYER-1):
    # 学習データに対するモデル出力を計算
    yhat = model(data.input[:bound], num_used_layers=i+2)

    fitrate = fit(yhat.detach().cpu(), data.output[:bound].detach().cpu())

    plt.figure()
    plt.plot(data.time[:bound].detach().cpu().numpy(), data.output[:bound].detach().cpu().numpy(), '-k', label='True value')
    plt.plot(data.time[:bound].detach().cpu().numpy(), yhat.detach().cpu().numpy(), '-.r', label=f'Model output ({fitrate:.1f} %)')
    plt.grid()
    plt.xlabel('Time k')
    plt.ylabel('Output y')
    plt.legend()
    plt.xlim(0, 100)
    plt.savefig(f'{FIGURE_FOLDER}pruned/fig_train_zoom_{i}.png')


    # 検証データに対するモデル出力を計算
    yhat = model(data.input[bound:bound+20000], num_used_layers=i+2)

    fitrate = fit(yhat.detach().cpu(), data.output[bound:bound+20000].detach().cpu())

    plt.figure()
    plt.plot(data.time[bound:bound+20000].detach().cpu().numpy(), data.output[bound:bound+20000].detach().cpu().numpy(), '-k', label='True value')
    plt.plot(data.time[bound:bound+20000].detach().cpu().numpy(), yhat.detach().cpu().numpy(), '-.r', label=f'Model output ({fitrate:.1f} %)')
    plt.grid()
    plt.xlabel('Time k')
    plt.ylabel('Output y')
    plt.legend()
    plt.xlim(bound, bound+100)
    plt.savefig(f'{FIGURE_FOLDER}pruned/fig_validation_zoom_{i}.png')


    # 枝刈り後の階層的NNをプロット
    w, _intermediate_w = model.net(z, num_used_layers=i+2)

    plt.figure()
    plt.plot(z.detach().cpu().numpy(), w.detach().cpu().numpy(), '-r')
    plt.grid()
    plt.xlabel('z')
    plt.ylabel('w')
    plt.ylim(w_min - (w_max - w_min)*0.1, w_max + (w_max - w_min)*0.1)
    plt.legend()
    plt.savefig(f'{FIGURE_FOLDER}pruned/fig_function_{i}.png')


    # 適合率を計算
    fitrate = fit(yhat.detach().cpu(), data.output[bound:bound+20000].detach().cpu())
    rmse_hat = rmse(yhat.detach().cpu(), data.output[bound:bound+20000].detach().cpu()).item()
    print(f'pruned: {amounts[i]*100} %, number of parameters: {params[i]},  fit: {fitrate}, rmse: {rmse_hat}')

    amount_log.append(amounts[i])
    params_log.append(params[i])
    fit_log.append(fitrate)
    rmse_log.append(rmse_hat)


# ログを保存
result = np.array([amount_log, params_log, fit_log, rmse_log]).transpose()
np.savetxt(outfile_path, result, delimiter=',')


# LTIモデルのパラメータを表示
print(f'A: {model.A.weight}')
print(f'Bu: {model.Bu.weight}')
print(f'Bw: {model.Bw.weight}')
print(f'Cy: {model.Cy.weight}')
print(f'Cz: {model.Cz.weight}')
print(f'Dyu: {model.Dyu.weight}')
print(f'Dyw: {model.Dyw.weight}')
print(f'Dzu: {model.Dzu.weight}')
# print(f'Dzw: {model.Dzw.weight}')

plt.show()
