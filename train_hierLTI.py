#!/usr/bin/env python
# coding: utf-8

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from datetime import date

from MyNets.lti import HierarchicalLTI2, HierarchicalLTIwithAllInput
from MyUtilities.my_dataloader import InOutDataSilverbox
from MyUtilities.try_gpu import try_gpu
from MyUtilities.fit import fit
from MyUtilities.measure_elapsed_time import tic, toc


# 階層的な線形状態空間モデルの学習を行う

DATAPATH = 'data/csv/2022-10-05_oscil100'

DATANAME_TRAIN = 'inoutdata_multimodes.csv'
DATANAME_VAL = 'inoutdata_multimodes.csv'

PATH_TRAIN = DATAPATH + '/' + DATANAME_TRAIN
PATH_VAL = DATAPATH + '/' + DATANAME_VAL

LENGTH = 3000    # シミュレーションの長さ

NUM_LAYER = 4

name = date.today().strftime('%Y-%m-%d') + '-' + str(NUM_LAYER) + '_4_oscil100_3000_losstime10'
FIGUREFOLDER = 'figures/HierLTI/' + name + '/'
if not os.path.exists(FIGUREFOLDER):
        # ディレクトリが存在しない場合、ディレクトリを作成する
        os.makedirs(FIGUREFOLDER)

MODEL_NAME = 'HierLTI_' + name + '.pth'
MODEL_PATH = 'TrainedModels/' + MODEL_NAME

EPOCHS = 20000

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# データ読み込み
data_train = InOutDataSilverbox(PATH_TRAIN)
data_val = InOutDataSilverbox(PATH_VAL)

data_train.input = data_train.input[:9000]
data_train.output = data_train.output[:9000]
data_train.time = torch.arange(0, len(data_train.input), 1) # 離散時刻
data_train.N = len(data_train.input)

data_val.input = data_val.input[9000:]
data_val.output = data_val.output[9000:]
data_val.time = torch.arange(0, len(data_val.input), 1)


# モデル定義
# model = HierarchicalLTIwithAllInput(num_layer=NUM_LAYER, u_size=1, y_size=1, order_of_a_LTImodel=2)
model = HierarchicalLTI2(num_layer=NUM_LAYER, u_size=1, y_size=1, order_of_a_LTImodel=2)
model.initialize_A(0.8)
model = try_gpu(model)


# 学習
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

loss_history = []
term_number_history = []
    
regularization_parameter = [1.0 for i in range(1, NUM_LAYER+1)]
probability = [gamma / sum(regularization_parameter) for gamma in regularization_parameter] # 確率分布 p(l) = γ_l

losstime = 10

tic()   # 経過時間測定開始
for epoch in range(EPOCHS):
    optimizer.zero_grad()

    term_number = np.random.choice(list(range(1, NUM_LAYER+1)), p=probability)      # 今ステップで使う層数 l を，確率分布 p(l) = γ_l に基いてランダムに選ぶ
    # term_number = 8
    start_time = np.random.choice(data_train.N - LENGTH)                            # 今ステップにおけるデータの切り出し始めの位置を一様分布からサンプリング

    yhat = model(data_train.input[start_time:start_time+LENGTH], num_used_layers=term_number)

    loss = criterion(yhat[losstime:], data_train.output[start_time+losstime:start_time+LENGTH])

    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print('epoch: ', epoch, ' loss: ', loss.item())

    loss_history.append(loss.item())
    term_number_history.append(term_number)

toc()   # 経過時間測定終了

yhat = model(data_train.input[start_time:start_time+LENGTH], num_used_layers=NUM_LAYER)
print(model.A)
print(model.B)
print(model.C)
print(model.D)

# 学習データと検証データに対する適合率を計算
yhat_train = model(data_train.input[:LENGTH])
yhat_val = model(data_val.input[:LENGTH])

fit_train = fit(yhat_train.detach().cpu(), data_train.output[:LENGTH].detach().cpu())
fit_val = fit(yhat_val.detach().cpu(), data_val.output[:LENGTH].detach().cpu())


torch.save(model.to('cpu').state_dict(), MODEL_PATH)    # モデルを保存


# 各層のモデルに対する平均二乗誤差の変化をプロット
plt.figure()
history = np.array([list(range(EPOCHS)), term_number_history, loss_history]).transpose()
for i in range(1, NUM_LAYER+1):
    temp = history[history[:, 1] == i]
    plt.plot(temp[:, 0], temp[:, 2], label=f'l = {i}')
# plt.plot(loss_history)
plt.yscale('log')
plt.grid()
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.legend()
plt.savefig(FIGUREFOLDER + 'fig_loss.png')

# 学習データに対してモデル出力をプロット
plt.figure()
plt.plot(data_train.time[:LENGTH], data_train.output[:LENGTH].detach().cpu(), '-k', label='Training data')
plt.plot(data_train.time[:LENGTH], yhat_train.squeeze().detach().cpu(), '-.r', label='Model output (Fit: '+str(round(fit_train, 2))+' %)')
plt.grid()
# plt.xlabel('Time [s]')
# plt.ylabel('Velocity [km/h]')
# plt.ylim(-10, 80)
plt.legend()
plt.savefig(FIGUREFOLDER + 'fig_output_train.png')

# 検証データに対してモデル出力をプロット
plt.figure()
plt.plot(data_val.time[:LENGTH], data_val.output[:LENGTH].detach().cpu(), '-k', label='Validation data')
plt.plot(data_val.time[:LENGTH], yhat_val.squeeze().detach().cpu(), '-.r', label='Model output (Fit: '+str(round(fit_val, 2))+' %)')
plt.grid()
# plt.xlabel('Time [s]')
# plt.ylabel('Velocity [km/h]')
# plt.ylim(-10, 80)
plt.legend()
plt.savefig(FIGUREFOLDER + 'fig_output_validation.png')


plt.show()


