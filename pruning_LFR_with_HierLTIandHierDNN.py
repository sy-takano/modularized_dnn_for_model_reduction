#!/usr/bin/env python
# coding: utf-8

import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import matplotlib.pyplot as plt

from MyNets.rnn_lfr import HierarchicalNeuralLFR

from MyUtilities.my_dataloader import InOutData, InOutDataSilverbox
from MyUtilities.try_gpu import try_gpu
from MyUtilities.measure_elapsed_time import tic, toc
from MyUtilities.rmse import rmse
from MyUtilities.fit import fit

from scipy import signal

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


# 階層的LTIと階層的DNNを組み合わせた非線形LFRモデルの枝刈りを行う

# データのパスを指定
DATA_PATH= 'data/csv/WienerHammerBenchmark.csv'
# DATA_PATH= 'data/csv/SNLS80mV.csv'

# モデル読み込み
date = '2023-02-05-WHB_zwsize9'
# date = '2022-12-13'
FIGURE_FOLDER = './figures/LFR_with_HierLTIandHierDNN/' + date + '/'
model_path = './TrainedModels/LFR_with_HierLTIandDNN_' + date + '.pth'

outfile_path = FIGURE_FOLDER + 'accuracy.csv'

HIDDEN_SIZE = 64
X_SIZE = 2
INPUT_SIZE = 1
OUTPUT_SIZE = 1
NUM_LAYER_LTI = 4
NUM_LAYER_NN = 5

model = HierarchicalNeuralLFR(num_layer_NN=NUM_LAYER_NN, hidden_size=64, num_layer_LTI=NUM_LAYER_LTI, u_size=1, y_size=1, z_size=9, w_size=9)
model = try_gpu(model)
model.A, model.B, model.C, model.D = model.make_ABCDmatrices(NUM_LAYER_LTI)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.A, model.B, model.C, model.D = model.make_ABCDmatrices(NUM_LAYER_LTI)



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
for i in range(2, NUM_LAYER_LTI):
    params.append(params[-1] + HIDDEN_SIZE*HIDDEN_SIZE + HIDDEN_SIZE + HIDDEN_SIZE*OUTPUT_SIZE + OUTPUT_SIZE)
params = np.array(params) + params_LTI
amounts = 1 - params/params_sum
print(params)


# データ読み込み
data = InOutDataSilverbox(DATA_PATH)
data.input = data.input[50000:]     # マルチサイン入力の部分のみ使う
data.output = data.output[50000:]

data.time = torch.arange(0, len(data.input), 1)


# # 学習されたNNを学習データの範囲でプロット
bound = 20000   # はじめ20000データは学習データ
# yhat = model(data.input[:bound], num_used_layers_LTI=NUM_LAYER_LTI, num_used_layers_NN=NUM_LAYER_NN)

# N = 10000
# z = torch.linspace(start=min(model.z_record).item(), end=max(model.z_record).item(), steps=N, device=device).reshape(-1, 1)
# w, _intermediate = model.net(z)

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
num_used_layers_LTI_log = []
num_used_layers_NN_log = []
amount_log = []
params_log = []
fit_log = []
rmse_log = []
etime = []
# f = []
# Pxx = []
for i in range(1, NUM_LAYER_LTI+1):
# for i in [1, 4]:
    for jj in range(2, NUM_LAYER_NN+1):
    # for jj in [2, 5]:
        # 学習データに対するモデル出力を計算
        yhat = model(data.input[:bound], num_used_layers_LTI=i, num_used_layers_NN=jj)

        fitrate = fit(yhat.detach().cpu(), data.output[:bound].detach().cpu())

        plt.figure()
        plt.plot(data.time[:bound].detach().cpu().numpy(), data.output[:bound].detach().cpu().numpy(), '-k', label='True value')
        plt.plot(data.time[:bound].detach().cpu().numpy(), yhat.detach().cpu().numpy(), '-.r', label=f'Model output ({fitrate:.1f} %)')
        plt.grid()
        plt.xlabel('Time k')
        plt.ylabel('Output y')
        # plt.legend()
        plt.xlim(0, 1000)
        plt.savefig(f'{FIGURE_FOLDER}pruned/fig_train_zoom_nofit_{i}_{jj}.png')


        # 検証データに対するモデル出力を計算
        tic()
        yhat = model(data.input[bound:bound+20000], num_used_layers_LTI=i, num_used_layers_NN=jj)
        et = toc()

        fitrate = fit(yhat.detach().cpu(), data.output[bound:bound+20000].detach().cpu())

        plt.figure()
        # plt.plot(data.time[bound:bound+20000].detach().cpu().numpy(), data.output[bound:bound+20000].detach().cpu().numpy(), '-k', label='True value')
        # plt.plot(data.time[bound:bound+20000].detach().cpu().numpy(), yhat.detach().cpu().numpy(), '-.r', label=f'Model output ({fitrate:.1f} %)')
        plt.plot(data.time[:20000].detach().cpu().numpy(), data.output[bound:bound+20000].detach().cpu().numpy(), '-k', label='True value')
        plt.plot(data.time[:20000].detach().cpu().numpy(), yhat.detach().cpu().numpy(), '-.r', label=f'Model output ({fitrate:.1f} %)')
        plt.grid()
        plt.xlabel('Time k')
        plt.ylabel('Output y')
        # plt.legend()
        # plt.xlim(bound, bound+1000)
        plt.xlim(0, 1000)
        plt.ylim(-1.0, 0.75)
        # plt.savefig(f'{FIGURE_FOLDER}pruned/fig_validation_data.png')
        plt.savefig(f'{FIGURE_FOLDER}pruned/fig_validation_zoom_nofit_{i}_{jj}.png')


        # 適合率を計算
        fitrate = fit(yhat.detach().cpu(), data.output[bound:bound+20000].detach().cpu())
        rmse_hat = rmse(yhat.detach().cpu(), data.output[bound:bound+20000].detach().cpu()).item()
        # print(f'pruned: {amounts[i-1]*100} %, number of parameters: {params[i-1]},  fit: {fitrate}, rmse: {rmse_hat}')
        print(f'Number of LTI modules: {i}, number of DNN modules: {jj-1}, fit: {fitrate}, rmse: {rmse_hat}')

        amount_log.append(amounts[i-1])
        params_log.append(params[i-1])
        fit_log.append(fitrate)
        rmse_log.append(rmse_hat)
        num_used_layers_LTI_log.append(i)
        num_used_layers_NN_log.append(jj-1)
        etime.append(et)


        # err = yhat.detach().cpu() - data.output[bound:bound+20000].detach().cpu()

        # plt.figure()
        # plt.plot(err)


        # plt.figure()
        # f, Pxx_den = signal.periodogram(err)
        # plt.loglog(f, Pxx_den)
        # # plt.ylim([1e-7, 1e2])
        # plt.xlabel('frequency [Hz]')
        # plt.ylabel('PSD [V**2/Hz]')
    
plt.show()


# ログを保存
result = np.array([amount_log, params_log, fit_log, rmse_log, num_used_layers_LTI_log, num_used_layers_NN_log, etime]).transpose()
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
