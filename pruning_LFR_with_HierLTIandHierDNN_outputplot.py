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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


# 階層的LTIと階層的DNNを組み合わせた非線形LFRモデルの枝刈りを行ない，検証データに対する出力を比較
# それぞれの計算時間も計測


# データのパスを指定
DATA_PATH= 'data/csv/WienerHammerBenchmark.csv'
# DATA_PATH= 'data/csv/SNLS80mV.csv'

# モデル読み込み
date = '2023-02-05-WHB_zwsize9'
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




# データ読み込み
data = InOutDataSilverbox(DATA_PATH)
data.input = data.input[50000:]     # マルチサイン入力の部分のみ使う
data.output = data.output[50000:]

data.time = torch.arange(0, len(data.input), 1)

bound = 20000   # はじめ20000データは学習データ


# 検証データに対するモデル出力を計算

plt.figure()
plt.grid()
plt.xlabel('Time k')
plt.ylabel('Output y')
# plt.legend()
plt.xlim(0, 500)
plt.ylim(-1.0, 0.75)

plt.plot(data.time[:20000].detach().cpu().numpy(), data.output[bound:bound+20000].detach().cpu().numpy(), '-k', label='True value')

plt.savefig(f'{FIGURE_FOLDER}pruned/fig_validation_comparison_data.png')

tic()
yhat = model(data.input[bound:bound+20000], num_used_layers_LTI=4, num_used_layers_NN=5)
toc()
fitrate = fit(yhat.detach().cpu(), data.output[bound:bound+20000].detach().cpu())
plt.plot(data.time[:20000].detach().cpu().numpy(), yhat.detach().cpu().numpy(), '--r', label=f'Model output ({fitrate:.1f} %)')

print(f'RMSE (n=4, m=4): {rmse(yhat.detach().cpu(), data.output[bound:bound+20000].detach().cpu()).item()}')

plt.savefig(f'{FIGURE_FOLDER}pruned/fig_validation_comparison_4_5.png')

tic()
yhat = model(data.input[bound:bound+20000], num_used_layers_LTI=1, num_used_layers_NN=2)
toc()
fitrate = fit(yhat.detach().cpu(), data.output[bound:bound+20000].detach().cpu())
plt.plot(data.time[:20000].detach().cpu().numpy(), yhat.detach().cpu().numpy(), '-.b', label=f'Model output ({fitrate:.1f} %)')

print(f'RMSE (n=1, m=1): {rmse(yhat.detach().cpu(), data.output[bound:bound+20000].detach().cpu()).item()}')

plt.savefig(f'{FIGURE_FOLDER}pruned/fig_validation_comparison_4_5_and_1_2.png')

plt.show()
