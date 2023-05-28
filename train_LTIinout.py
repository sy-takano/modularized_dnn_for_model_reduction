#!/usr/bin/env python
# coding: utf-8

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from datetime import date

from MyNets.lti_inout import FIR, ARX
from MyUtilities.my_dataloader import InOutDataSilverbox
from MyUtilities.try_gpu import try_gpu
from MyUtilities.fit import fit
from MyUtilities.measure_elapsed_time import tic, toc

# 入出力表現された線形モデルの学習

DATAPATH = 'data/csv'

DATANAME_TRAIN = 'SNLS80mV.csv'
DATANAME_VAL = 'SNLS80mV.csv'

PATH_TRAIN = DATAPATH + '/' + DATANAME_TRAIN
PATH_VAL = DATAPATH + '/' + DATANAME_VAL

LENGTH = 300    # シミュレーションの長さ

FIGUREFOLDER = 'figures/ARX/' + date.today().strftime('%Y-%m-%d') + '/'
if not os.path.exists(FIGUREFOLDER):
        # ディレクトリが存在しない場合、ディレクトリを作成する
        os.makedirs(FIGUREFOLDER)

MODEL_NAME = 'ARX_' + date.today().strftime('%Y-%m-%d') + '.pth'
MODEL_PATH = 'TrainedModels/' + MODEL_NAME

EPOCHS = 5000

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


data_train = InOutDataSilverbox(PATH_TRAIN)
data_val = InOutDataSilverbox(PATH_VAL)

data_train.input = data_train.input[50000:53000]     # マルチサイン入力の部分のみ使用する
data_train.output = data_train.output[50000:53000]

data_val.input = data_val.input[70000:73000]     # マルチサイン入力の部分のみ使用する
data_val.output = data_val.output[70000:73000]

order = 3
# model = FIR(order, 1, 1)
model = ARX(order, order, 1, 1)



model = try_gpu(model)


criterion = torch.nn.MSELoss()
# optimizer = torch.optim.LBFGS(model.parameters(), max_iter=1000, lr=0.1, tolerance_grad=1e-12, tolerance_change=0)
optimizer = torch.optim.Adam(model.parameters())

loss_history = []

tic()   # 経過時間測定開始
for epoch in range(EPOCHS):
    optimizer.zero_grad()

    yhat = model(data_train.input)

    loss = criterion(yhat, data_train.output[order+1:])

    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print('epoch: ', epoch, ' loss: ', loss.item())

    loss_history.append(loss.item())

toc()   # 経過時間測定終了

# 学習データと検証データに対する適合率を計算
yhat_train = model(data_train.input[:LENGTH])
yhat_val = model(data_val.input[:LENGTH])

fit_train = fit(yhat_train.detach().cpu(), data_train.output[order+1:LENGTH].detach().cpu())
fit_val = fit(yhat_val.detach().cpu(), data_val.output[order+1:LENGTH].detach().cpu())


torch.save(model.to('cpu').state_dict(), MODEL_PATH)    # モデルを保存


# 平均二乗誤差の変化をプロット
plt.figure()
plt.plot(loss_history)
plt.yscale('log')
plt.grid()
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.legend()
plt.savefig(FIGUREFOLDER + 'fig_loss.png')


# 学習データに対してモデル出力をプロット
plt.figure()
plt.plot(data_train.output[order+1:LENGTH].detach().cpu(), '-k', label='Training data')
plt.plot(yhat_train.squeeze().detach().cpu(), '-.r', label='Model output (Fit: '+str(round(fit_train, 2))+' %)')

plt.grid()
plt.legend()
plt.savefig(FIGUREFOLDER + 'fig_output_train.png')

# 検証データに対してモデル出力をプロット
plt.figure()
plt.plot(data_val.output[order+1:LENGTH].detach().cpu(), '-k', label='Validation data')
plt.plot(yhat_val.squeeze().detach().cpu(), '-.r', label='Model output (Fit: '+str(round(fit_val, 2))+' %)')

plt.grid()
# plt.xlabel('Time [s]')
# plt.ylabel('Velocity [km/h]')
# plt.ylim(-10, 80)
plt.legend()
plt.savefig(FIGUREFOLDER + 'fig_output_validation.png')


plt.show()

