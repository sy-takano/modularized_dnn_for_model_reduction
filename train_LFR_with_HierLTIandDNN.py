#!/usr/bin/env python
# coding: utf-8

import os
import sys

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from datetime import date

from MyNets.rnn_lfr import HierarchicalLTILFR
from MyUtilities.my_dataloader import InOutDataSilverbox
from MyUtilities.try_gpu import try_gpu
from MyUtilities.fit import fit
from MyUtilities.measure_elapsed_time import tic, toc
from MyUtilities.load_csv_as_tensor import load_csv_as_tensor

# 階層的LTIとFNNで構成した非線形LFRモデルの学習

DATAPATH = 'data/csv/silverbox'

DATANAME_TRAIN = 'SNLS80mV.csv'
DATANAME_VAL = 'SNLS80mV.csv'

PATH_TRAIN = DATAPATH + '/' + DATANAME_TRAIN
PATH_VAL = DATAPATH + '/' + DATANAME_VAL

LENGTH = 1000    # シミュレーションの長さ

FIGUREFOLDER = 'figures/LFR_with_HierLTIandDNN/' + date.today().strftime('%Y-%m-%d') + '/'
if not os.path.exists(FIGUREFOLDER):
        # ディレクトリが存在しない場合、ディレクトリを作成する
        os.makedirs(FIGUREFOLDER)

MODEL_NAME = 'LFR_with_HierLTIandDNN_' + date.today().strftime('%Y-%m-%d') + '.pth'
MODEL_PATH = 'TrainedModels/' + MODEL_NAME

EPOCHS = 20000
NUM_LAYER = 4

def initialize_net_parameter(parameter, range):
    nn.init.uniform_(parameter, -range, range)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


data_train = InOutDataSilverbox(PATH_TRAIN)
data_val = InOutDataSilverbox(PATH_VAL)

data_train.input = data_train.input[50000:]     # マルチサイン入力の部分のみ使用する
data_train.output = data_train.output[50000:]

data_val.input = data_val.input[60000:]     # マルチサイン入力の部分のみ使用する
data_val.output = data_val.output[60000:]

model = HierarchicalLTILFR(num_layer_NN=5, hidden_size=64, num_layer_LTI=NUM_LAYER, u_size=1, y_size=1)

# モデルパラメータの初期化
A_ini = load_csv_as_tensor(DATAPATH + '/A_ini.csv')
B_ini = load_csv_as_tensor(DATAPATH + '/B_ini.csv').unsqueeze(1)
C_ini = load_csv_as_tensor(DATAPATH + '/C_ini.csv').unsqueeze(0)
D_ini = load_csv_as_tensor(DATAPATH + '/D_ini.csv').unsqueeze(0).unsqueeze(1)

model.lti_list[0].B.weight = nn.Parameter(torch.cat((B_ini[0:2, :], model.lti_list[0].B.weight[:, 1].unsqueeze(1)), 1))
# model.lti_list[0].D.weight = nn.Parameter(D_ini)  # D=0の場合はコメントアウトのままでOK

for i in range(NUM_LAYER):
    model.lti_list[i].A.weight = nn.Parameter(A_ini[2*i:2*(i+1), 2*i:2*(i+1)])
    model.lti_list[i].C.weight = nn.Parameter(torch.cat((C_ini[:, 2*i:2*(i+1)], model.lti_list[i].C.weight[1, :].unsqueeze(0)), 0))

    if i>0:
        model.lti_list[i].B.weight = nn.Parameter(A_ini[2*i:2*(i+1), 2*(i-1):2*i])


model = try_gpu(model)

criterion = torch.nn.MSELoss()
# optimizer = torch.optim.LBFGS(model.parameters(), max_iter=1000, lr=0.1, tolerance_grad=1e-12, tolerance_change=0)
optimizer = torch.optim.Adam(model.parameters())

loss_history = []
term_number_history = []
    
regularization_parameter = [1.0 for i in range(1, NUM_LAYER+1)]
probability = [gamma / sum(regularization_parameter) for gamma in regularization_parameter] # 確率分布 p(l) = γ_l


tic()   # 経過時間測定開始
for epoch in range(EPOCHS):
    optimizer.zero_grad()

    term_number = np.random.choice(list(range(1, NUM_LAYER+1)), p=probability)    # 今ステップで使う層数 l を，確率分布 p(l) = γ_l に基いてランダムに選ぶ
    # start_time = np.random.choice(20000)                            # 今ステップにおけるデータの切り出し始めの位置を一様分布からサンプリング
    start_time = 0

    yhat = model(data_train.input[start_time:start_time+LENGTH], num_used_layers=term_number)

    loss = criterion(yhat, data_train.output[start_time:start_time+LENGTH])

    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print('epoch: ', epoch, ' loss: ', loss.item())

    loss_history.append(loss.item())
    term_number_history.append(term_number)


toc()   # 経過時間測定終了

# 学習データと検証データに対する適合率を計算
yhat_train = model(data_train.input[:LENGTH])
yhat_val = model(data_val.input[:LENGTH])

fit_train = fit(yhat_train.detach().cpu(), data_train.output[:LENGTH].detach().cpu())
fit_val = fit(yhat_val.detach().cpu(), data_val.output[:LENGTH].detach().cpu())


torch.save(model.to('cpu').state_dict(), MODEL_PATH)    # モデルを保存

log = np.array([term_number_history, loss_history]).transpose()
np.savetxt(FIGUREFOLDER + 'log.csv', log, delimiter=',')

# 各層のモデルに対する平均二乗誤差の変化をプロット
plt.figure()
history = np.array([list(range(EPOCHS)), term_number_history, loss_history]).transpose()
for i in range(1, NUM_LAYER+1):
    temp = history[history[:, 1] == i]
    plt.plot(temp[:, 0], temp[:, 2], label=f'term number = {i}')
# plt.plot(loss_history)
plt.yscale('log')
plt.grid()
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.legend()
plt.savefig(FIGUREFOLDER + 'fig_loss.png')


# # 学習データに対してモデル出力をプロット
plt.figure()
plt.plot(data_train.output[:LENGTH].detach().cpu(), '-k', label='Training data')
plt.plot(yhat_train.squeeze().detach().cpu(), '-.r', label='Model output (Fit: '+str(round(fit_train, 2))+' %)')

plt.grid()
# plt.xlabel('Time [s]')
# plt.ylabel('Velocity [km/h]')
# plt.ylim(-10, 80)
plt.legend()
plt.savefig(FIGUREFOLDER + 'fig_RNNLFR_train.png')

# # 検証データに対してモデル出力をプロット
plt.figure()
plt.plot(data_val.output[:LENGTH].detach().cpu(), '-k', label='Validation data')
plt.plot(yhat_val.squeeze().detach().cpu(), '-.r', label='Model output (Fit: '+str(round(fit_val, 2))+' %)')

plt.grid()
# plt.xlabel('Time [s]')
# plt.ylabel('Velocity [km/h]')
# plt.ylim(-10, 80)
plt.legend()
plt.savefig(FIGUREFOLDER + 'fig_RNNLFR_validation.png')


plt.show()


