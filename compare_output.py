#!/usr/bin/env python
# coding: utf-8

import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import matplotlib.pyplot as plt

from MyNets.rnn_lfr import HierarichicalRNNLFR, DNNLFR
from MyNets.dnn import DeepNeuralNetwork

from MyUtilities.my_dataloader import InOutData, InOutDataSilverbox
from MyUtilities.try_gpu import try_gpu
from MyUtilities.measure_elapsed_time import tic, toc
from MyUtilities.rmse import rmse
from MyUtilities.fit import fit


# 学習済みの階層的NNとFNNの出力を計算して比較する


DATAPATH = 'data/csv'

DATANAME_TRAIN = 'SNLS80mV.csv'
DATANAME_VAL = 'SNLS80mV.csv'

PATH_TRAIN = DATAPATH + '/' + DATANAME_TRAIN
PATH_VAL = DATAPATH + '/' + DATANAME_VAL


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# 5層の学習済み階層的NN
date = '2022-09-03'
model_path = './TrainedModels/LFR_with_hierDNN_' + date + '.pth'
model = HierarichicalRNNLFR(hidden_size=64, num_layer=5, x_size=2)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model = try_gpu(model)

# # 17層の学習済み階層的NN
# date_2 = '2022-01-28'
# model_path_2 = './TrainedModels/LFR_with_hierDNN_' + date_2 + '.pth'
# model_2 = HierarichicalRNNLFR(hidden_size=23, num_layer=17, x_size=2)
# model_2.load_state_dict(torch.load(model_path_2, map_location=torch.device('cpu')))
# model_2 = try_gpu(model_2)

# 5層の学習済みFNN
date_DNN = '2022-09-05'
model_path_DNN = './TrainedModels/LFR_with_DNN_' + date_DNN + '.pth'
model_DNN = DNNLFR(hidden_size=64, num_layer=5, x_size=2)
model_DNN.load_state_dict(torch.load(model_path_DNN, map_location=torch.device('cpu')))
model_DNN = try_gpu(model_DNN)


FIGURE_FOLDER = './figures/LFR_with_hierDNN/' + date + '/'


data = InOutDataSilverbox(PATH_TRAIN)

# データプロット
fig = plt.figure(tight_layout=True, figsize=[6.4, 3.5])

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams["font.family"] = "Times New Roman" 
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.size"] = 15

ax1 = fig.add_subplot(2, 1, 1)
ax1.tick_params(labelbottom=False)
ax2 = fig.add_subplot(2, 1, 2)

ax1.plot(np.arange(0, len(data.input.detach().cpu().numpy()), 1), data.input.detach().cpu().numpy())
ax1.set_ylabel("Input $u$")
ax1.grid()
ax1.set_xlim(0, 120000)

ax2.plot(np.arange(0, len(data.input.detach().cpu().numpy()), 1), data.output.detach().cpu().numpy())
ax2.set_ylabel("Output $y$")
ax2.set_xlabel("Time $k$")
ax2.grid()
ax2.set_xlim(0, 120000)

plt.show()


data.input = data.input[50000:] # マルチサイン入力の部分のみ使う
data.output = data.output[50000:]

data.time = torch.arange(0, len(data.input), 1)


for i in [0]:
    bound = 20000   # 学習データと検証データの境界
    length = 300    # シミュレーションを行う長さ
    shift = 50

    # # 学習データに対するモデル出力を計算してプロット
    # yhat = model(data.input[:bound], num_used_layers=i+1)

    # fitrate = fit(yhat.detach().cpu(), data.output[:bound].detach().cpu())

    # plt.figure()
    # plt.plot(data.time[:bound].detach().cpu().numpy(), data.output[:bound].detach().cpu().numpy(), '-k', label='True value')
    # plt.plot(data.time[:bound].detach().cpu().numpy(), yhat.detach().cpu().numpy(), '-.r', label=f'Model output ({fitrate:.1f} %)')
    # plt.grid()
    # plt.xlabel('Time k')
    # plt.ylabel('Output y')
    # plt.legend()
    # plt.xlim(0, 100)
    # plt.savefig(f'{FIGURE_FOLDER}pruned/fig_train_zoom_{i}.png')


    # 検証データに対するモデル出力を計算してプロット
    yhat = model(data.input[bound:bound+length], num_used_layers=4)
    # yhat_2 = model_2(data.input[bound:bound+length], num_used_layers=16)
    yhat_DNN = model_DNN(data.input[bound:bound+length])

    # fitrate = fit(yhat.detach().cpu(), data.output[bound:bound+20000].detach().cpu())
    # fitrate_2 = fit(yhat_2.detach().cpu(), data.output[bound:bound+20000].detach().cpu())
    # fitrate_DNN = fit(yhat_DNN.detach().cpu(), data.output[bound:bound+20000].detach().cpu())

    

    # 出力をプロット
    plt.figure(tight_layout=True, figsize=[6.4, 3.5])
    
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams["font.family"] = "Times New Roman" 
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["font.size"] = 15
    
    plt.plot(data.time[bound:bound+length].detach().cpu().numpy() - bound - 50, data.output[bound:bound+length].detach().cpu().numpy(), '-k', label='True value')
    plt.plot(data.time[bound:bound+length].detach().cpu().numpy() - bound - 50, yhat_DNN.detach().cpu().numpy(), '-.b', label=f'Pruned FNN (5 layers, 64 nodes)')
    # plt.plot(data.time[bound:bound+length].detach().cpu().numpy() - bound, yhat_2.detach().cpu().numpy(), ':', color='magenta', label=f'Proposed model (17 layers, 23 nodes)')
    plt.plot(data.time[bound:bound+length].detach().cpu().numpy() - bound - 50, yhat.detach().cpu().numpy(), '--', color='red', label=f'Proposed model (5 layers, 64 nodes)')

    plt.grid()
    plt.xlabel('Time $k$')
    plt.ylabel('Output')
    # plt.legend()
    plt.xlim(0, 200)
    plt.savefig(f'{FIGURE_FOLDER}fig_comparison_output_zoom.png')
    plt.savefig(f'{FIGURE_FOLDER}fig_comparison_output_zoom.pdf')
    
    
    # 誤差をプロット
    plt.figure(tight_layout=True, figsize=[6.4, 3.5])
    
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams["font.family"] = "Times New Roman" 
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["font.size"] = 15

    # plt.plot(data.time[bound:bound+length].detach().cpu().numpy() - bound, data.output[bound:bound+length].detach().cpu().numpy() - yhat_2.detach().cpu().numpy(), '-.', color='m', label=f'Proposed model (17 layers, 23 nodes)')
    plt.plot(data.time[bound:bound+length].detach().cpu().numpy() - bound, data.output[bound:bound+length].detach().cpu().numpy() - yhat_DNN.detach().cpu().numpy(), '--', color='blue', label=f'Pruned FNN (5 layers, 64 nodes)')
    plt.plot(data.time[bound:bound+length].detach().cpu().numpy() - bound, data.output[bound:bound+length].detach().cpu().numpy() - yhat.detach().cpu().numpy(), '-', color='red', label=f'Proposed model (5 layers, 64 nodes)')

    plt.grid()
    plt.xlabel('Time $k$')
    plt.ylabel('Error $y(k) - \hat{y}(k)$')
    plt.legend()
    plt.xlim(0, length)
    plt.savefig(f'{FIGURE_FOLDER}fig_comparison_error_zoom.png')
    plt.savefig(f'{FIGURE_FOLDER}fig_comparison_error_zoom.pdf')



plt.show()
