#!/usr/bin/env python
# coding: utf-8

import os
import sys

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from datetime import date

from MyNets.rnn_lfr import HierarchicalNeuralLFR
from MyUtilities.my_dataloader import InOutDataSilverbox
from MyUtilities.try_gpu import try_gpu
from MyUtilities.fit import fit
from MyUtilities.measure_elapsed_time import tic, toc
from MyUtilities.load_csv_as_tensor import load_csv_as_tensor

# 階層的LTIと階層的DNNで構成した非線形LFRモデルの学習

# DATAPATH = 'data/csv/silverbox'
# DATAPATH = 'data/csv/WienerHammerBenchmark'
DATAPATH = 'data/csv/drivetrain'

# DATANAME_TRAIN = 'SNLS80mV.csv'
# DATANAME_VAL = 'SNLS80mV.csv'
# DATANAME_TRAIN = 'WienerHammerBenchmark.csv'
# DATANAME_VAL = 'WienerHammerBenchmark.csv'
DATANAME_TRAIN = 'drivetraindata_WLTC.csv'
DATANAME_VAL = 'drivetraindata_JC08.csv'

PATH_TRAIN = DATAPATH + '/' + DATANAME_TRAIN
PATH_VAL = DATAPATH + '/' + DATANAME_VAL

LENGTH = 1800    # シミュレーションの長さ

FIGUREFOLDER = 'figures/LFR_with_HierLTIandHierDNN/' + date.today().strftime('%Y-%m-%d') + '-DT/'
if not os.path.exists(FIGUREFOLDER):
        # ディレクトリが存在しない場合、ディレクトリを作成する
        os.makedirs(FIGUREFOLDER)

MODEL_NAME = 'LFR_with_HierLTIandDNN_' + date.today().strftime('%Y-%m-%d') + '-DT.pth'
MODEL_PATH = 'TrainedModels/' + MODEL_NAME

EPOCHS = 10000
NUM_LAYER_LTI = 4
NUM_LAYER_NN = 5
z_size = NUM_LAYER_LTI*2 + 1
# z_size = 11

def initialize_net_parameter(parameter, range):
    nn.init.uniform_(parameter, -range, range)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


data_train = InOutDataSilverbox(PATH_TRAIN)
data_val = InOutDataSilverbox(PATH_VAL)

# data_train.input = data_train.input[50000:]     # マルチサイン入力の部分のみ使用する
# data_train.output = data_train.output[50000:]

# data_val.input = data_val.input[70000:]     # マルチサイン入力の部分のみ使用する
# data_val.output = data_val.output[70000:]

model = HierarchicalNeuralLFR(num_layer_NN=NUM_LAYER_NN, hidden_size=64, num_layer_LTI=NUM_LAYER_LTI, u_size=1, y_size=1, z_size=z_size, w_size=z_size)

# # モデルパラメータの初期化
# A_ini = load_csv_as_tensor(DATAPATH + '/A_ini.csv')
# B_ini = load_csv_as_tensor(DATAPATH + '/B_ini.csv').unsqueeze(1)
# C_ini = load_csv_as_tensor(DATAPATH + '/C_ini.csv').unsqueeze(0)
# D_ini = load_csv_as_tensor(DATAPATH + '/D_ini.csv').unsqueeze(0).unsqueeze(1)

# perturb = 1e-1

# B_r = perturb*model.lti_list[0].B.weight.detach()
# B_r[0:2, 0] = B_ini[0:2, 0]
# # model.lti_list[0].B.weight = nn.Parameter(torch.cat((B_ini[0:2, :], perturb*torch.ones_like(model.lti_list[0].B.weight[:, 1].unsqueeze(1))), 1))
# # model.lti_list[0].B.weight = nn.Parameter(torch.cat((B_ini[0:2, :], perturb*model.lti_list[0].B.weight[:, 1].unsqueeze(1)), 1))
# model.lti_list[0].B.weight = nn.Parameter(B_r)
# # model.lti_list[0].D.weight = nn.Parameter(D_ini)  # D=0の場合はコメントアウトのままでOK

# for i in range(NUM_LAYER_LTI):
#     model.lti_list[i].A.weight = nn.Parameter(A_ini[2*i:2*(i+1), 2*i:2*(i+1)])
#     if z_size == 1:
#         model.lti_list[i].C.weight = nn.Parameter(torch.cat((C_ini[:, 2*i:2*(i+1)], perturb*model.lti_list[i].C.weight[1, :].unsqueeze(0)), 0))
#     else:
#         model.lti_list[i].C.weight = nn.Parameter(torch.cat((C_ini[:, 2*i:2*(i+1)], perturb*model.lti_list[i].C.weight[1:, :]), 0))

#     if i>0:
#         model.lti_list[i].B.weight = nn.Parameter(A_ini[2*i:2*(i+1), 2*(i-1):2*i])

# for param in model.net.parameters():
#     nn.init.uniform_(param, -0.01, 0.01)    # ニューラルネットワーク部分の初期化


model = try_gpu(model)


criterion = torch.nn.MSELoss()
# optimizer = torch.optim.LBFGS(model.parameters(), max_iter=1000, lr=0.1, tolerance_grad=1e-12, tolerance_change=0)
# optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
optimizer = torch.optim.Adam(model.parameters())
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 100)

loss_history = []
term_number_history_LTI = []
term_number_history_NN = []
    
regularization_parameter_LTI = [1.0 for i in range(1, NUM_LAYER_LTI+1)]
probability_LTI = [gamma / sum(regularization_parameter_LTI) for gamma in regularization_parameter_LTI] # 確率分布 p(l) = γ_l
    
regularization_parameter_NN = [1.0 for i in range(2, NUM_LAYER_NN+1)]
probability_NN = [gamma / sum(regularization_parameter_NN) for gamma in regularization_parameter_NN] # 確率分布 p(l) = γ_l

tic()   # 経過時間測定開始
for epoch in range(EPOCHS):
    optimizer.zero_grad()

    term_number_LTI = np.random.choice(list(range(1, NUM_LAYER_LTI+1)), p=probability_LTI)    # 今ステップで使うLTIの層数をランダムに選ぶ
    term_number_NN = np.random.choice(list(range(2, NUM_LAYER_NN+1)), p=probability_NN)    # 今ステップで使うDNNの層数をランダムに選ぶ
    # start_time = np.random.choice(20000)                            # 今ステップにおけるデータの切り出し始めの位置を一様分布からサンプリング
    start_time = 0

    yhat = model(data_train.input[start_time:start_time+LENGTH], num_used_layers_LTI=term_number_LTI, num_used_layers_NN=term_number_NN)

    loss = criterion(yhat, data_train.output[start_time:start_time+LENGTH])

    loss.backward()
    optimizer.step()
    # scheduler.step()

    if epoch % 10 == 0:
        print('epoch: ', epoch, ' loss: ', loss.item())

    loss_history.append(loss.item())
    term_number_history_LTI.append(term_number_LTI)
    term_number_history_NN.append(term_number_NN)


toc()   # 経過時間測定終了

# 学習データと検証データに対する適合率を計算
# yhat_train = model(data_train.input[:LENGTH])
# yhat_val = model(data_val.input[:LENGTH])
yhat_train = model(data_train.input)
yhat_val = model(data_val.input)

# fit_train = fit(yhat_train.detach().cpu(), data_train.output[:LENGTH].detach().cpu())
# fit_val = fit(yhat_val.detach().cpu(), data_val.output[:LENGTH].detach().cpu())
fit_train = fit(yhat_train.detach().cpu(), data_train.output.detach().cpu())
fit_val = fit(yhat_val.detach().cpu(), data_val.output.detach().cpu())

torch.save(model.to('cpu').state_dict(), MODEL_PATH)    # モデルを保存

log = np.array([term_number_history_LTI, term_number_history_NN, loss_history]).transpose()
np.savetxt(FIGUREFOLDER + 'log.csv', log, delimiter=',')

# 各層のモデルに対する平均二乗誤差の変化をプロット
plt.figure()
history = np.array([list(range(EPOCHS)), term_number_history_LTI, term_number_history_NN, loss_history]).transpose()
for i in range(1, NUM_LAYER_LTI+1):
    for jj in range(2, NUM_LAYER_NN+1):
        temp = history[history[:, 1] == i]
        temp = temp[temp[:, 2] == jj]
        plt.plot(temp[:, 0], temp[:, 3], label=f'(l_LTI, l_NN) = ({i}, {jj-1})')
# plt.plot(loss_history)
plt.yscale('log')
plt.grid()
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.legend()
plt.savefig(FIGUREFOLDER + 'fig_loss.png')


# # 学習データに対してモデル出力をプロット
plt.figure()
# plt.plot(data_train.output[:LENGTH].detach().cpu(), '-k', label='Training data')
plt.plot(data_train.output.detach().cpu(), '-k', label='Training data')
plt.plot(yhat_train.squeeze().detach().cpu(), '-.r', label='Model output (Fit: '+str(round(fit_train, 2))+' %)')

plt.grid()
# plt.xlabel('Time [s]')
# plt.ylabel('Velocity [km/h]')
# plt.ylim(-10, 80)
plt.legend()
plt.savefig(FIGUREFOLDER + 'fig_RNNLFR_train.png')

# # 検証データに対してモデル出力をプロット
plt.figure()
# plt.plot(data_val.output[:LENGTH].detach().cpu(), '-k', label='Validation data')
plt.plot(data_val.output.detach().cpu(), '-k', label='Validation data')
plt.plot(yhat_val.squeeze().detach().cpu(), '-.r', label='Model output (Fit: '+str(round(fit_val, 2))+' %)')

plt.grid()
# plt.xlabel('Time [s]')
# plt.ylabel('Velocity [km/h]')
# plt.ylim(-10, 80)
plt.legend()
plt.savefig(FIGUREFOLDER + 'fig_RNNLFR_validation.png')


plt.show()


