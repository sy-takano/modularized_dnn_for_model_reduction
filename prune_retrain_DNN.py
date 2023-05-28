#!/usr/bin/env python
# coding: utf-8

import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import matplotlib.pyplot as plt
from copy import deepcopy

from MyNets.dnn import DeepNeuralNetwork

from MyUtilities.try_gpu import try_gpu
from MyUtilities.measure_elapsed_time import tic, toc
from MyUtilities.rmse import rmse
from MyUtilities.fit import fit
from MyUtilities.load_csv_as_tensor import load_csv_as_tensor
from MyUtilities.standardize import standardize


def static_system(u:torch.tensor):
    # 学習対象の静的システム
    # y = torch.sin(u) / u
    # y = u[:, 0] +2*u[:, 1]
    x1 = u[:, 0]
    x2 = u[:, 1]
    y = x1*torch.exp(-torch.pow(x1, 2)-torch.pow(x2,2))
    y = y.reshape(-1,1)
    # y = step(u)
    return y

def step(x):
  return 1.0 * (x >= 0.0)

# 学習済みのFNNの枝刈りを行なった後に再学習


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# date = '2022-04-25-17-23'
date = '2022-09-21-17-23-california'
# date = '2022-09-21-5-64-california'
FIGURE_FOLDER = './figures/DNN/' + date + '/retrained/'
model_path = './TrainedModels/DNN_california_' + date + '.pth'
outfile_path = FIGURE_FOLDER + 'accuracy.csv'

# モデル読み込み
INPUT_SIZE = 8
OUTPUT_SIZE = 1
NUM_LAYER = 17
HIDDEN_SIZE = 23
model = DeepNeuralNetwork(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, output_size=OUTPUT_SIZE, num_layer=NUM_LAYER, nonlinearity='relu')
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model = try_gpu(model)

# 総パラメータ数表示
params_sum = 0
for p in model.parameters():
    if p.requires_grad:
        params_sum += p.numel()
print(f'number of parameters: {params_sum}')

params = []
params.append(INPUT_SIZE*OUTPUT_SIZE + OUTPUT_SIZE)
params.append(params[-1] + INPUT_SIZE*HIDDEN_SIZE + HIDDEN_SIZE + HIDDEN_SIZE*OUTPUT_SIZE + OUTPUT_SIZE)
for i in range(1, NUM_LAYER-2):
    params.append(params[-1] + HIDDEN_SIZE*HIDDEN_SIZE + HIDDEN_SIZE + HIDDEN_SIZE*OUTPUT_SIZE + OUTPUT_SIZE)   # i+1層の階層的NNのパラメータ数

params = np.array(params)
amounts = 1 - params/params_sum
for i, amount in enumerate(amounts):
    if amount < 0:
        amounts[i] = 0

print(params)
print(amounts)

# データ読み込み/生成
DATAPATH = 'data/csv'
DATANAME_TRAIN = 'cadata.csv'
DATANAME_VAL = 'cadata.csv'
PATH_TRAIN = DATAPATH + '/' + DATANAME_TRAIN
PATH_VAL = DATAPATH + '/' + DATANAME_VAL
data_train = try_gpu(standardize(load_csv_as_tensor(PATH_TRAIN)))
data_val = try_gpu(standardize(load_csv_as_tensor(PATH_VAL)))

N_all = len(data_train)
np.random.seed(0)
randomize_index = np.random.permutation(N_all)
data_train = data_train[randomize_index]
data_val = data_val[randomize_index]

data_train = data_train[:15000]
data_val = data_val[15000:]

n_u = INPUT_SIZE
u = data_train[:, 0:n_u]
y = data_train[:, -1].reshape(-1,1)
u_val = data_val[:, 0:n_u]
y_val = data_val[:, -1].reshape(-1,1)
    
# N = 1000
# rng = 2
# u = 2*rng*(torch.rand(n_u*N, device=device) - 0.5).reshape(N, n_u)   # u ~ Uniform [-rng, rng)
# u_val = 2*rng*(torch.rand(n_u*N, device=device) - 0.5).reshape(N, n_u)   # u ~ Uniform [-rng, rng)
# # u_val = torch.linspace(start=-10, end=10, steps=N, device=device).reshape(-1, 1)   # u ~ Uniform [-10, 10)

# # システム定義
# y = static_system(u)
# y_val = static_system(u_val)



yhat_val = model(u_val)
print(rmse(yhat_val, y_val).item())


amount_log = []
params_log = []
fit_retrained_log = []
rmse_retrained_log = []
fit_before_retrained_log = []
rmse_before_retrained_log = []
EPOCHS = 10000

# amounts = np.concatenate([np.arange(0.00023, 0.1, 0.005), np.arange(0.1, 1.0+0.1, 0.1)], 0)
# # amounts = np.arange(0.0, 1.0+0.05, 0.05)
# amounts = 1.0 - amounts # 枝刈りするパラメータ数の割合
for idx, amount in enumerate(amounts):
    # モデル読み込み
    model = DeepNeuralNetwork(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, output_size=1, num_layer=NUM_LAYER, nonlinearity='relu')
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model = try_gpu(model)

    # 枝刈り対象のパラメータ
    # parameters_to_prune = (
    #     (model.fc_list[0], 'weight'),
    #     (model.fc_list[0], 'bias'),
    #     (model.fc_list[1], 'weight'),
    #     (model.fc_list[1], 'bias'),
    #     (model.fc_list[2], 'weight'),
    #     (model.fc_list[2], 'bias'),
    #     (model.fc_list[3], 'weight'),
    #     (model.fc_list[3], 'bias')
    # )
    parameters_to_prune = (
        (model.fc_list[0], 'weight'),
        (model.fc_list[0], 'bias'),
        (model.fc_list[1], 'weight'),
        (model.fc_list[1], 'bias'),
        (model.fc_list[2], 'weight'),
        (model.fc_list[2], 'bias'),
        (model.fc_list[3], 'weight'),
        (model.fc_list[3], 'bias'),
        (model.fc_list[4], 'weight'),
        (model.fc_list[4], 'bias'),
        (model.fc_list[5], 'weight'),
        (model.fc_list[5], 'bias'),
        (model.fc_list[6], 'weight'),
        (model.fc_list[6], 'bias'),
        (model.fc_list[7], 'weight'),
        (model.fc_list[7], 'bias'),
        (model.fc_list[8], 'weight'),
        (model.fc_list[8], 'bias'),
        (model.fc_list[9], 'weight'),
        (model.fc_list[9], 'bias'),
        (model.fc_list[10], 'weight'),
        (model.fc_list[10], 'bias'),
        (model.fc_list[11], 'weight'),
        (model.fc_list[11], 'bias'),
        (model.fc_list[12], 'weight'),
        (model.fc_list[12], 'bias'),
        (model.fc_list[13], 'weight'),
        (model.fc_list[13], 'bias'),
        (model.fc_list[14], 'weight'),
        (model.fc_list[14], 'bias'),
        (model.fc_list[15], 'weight'),
        (model.fc_list[15], 'bias'),
    )

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount
    )   # 絶対値の小さいパラメータから削除
    print(f'# of removed parameter : {params_sum-params[idx]}')

    yhat_val = model(u_val)
    fitrate_before_retrained = fit(yhat_val, y_val)
    rmse_hat_before_retrained = rmse(yhat_val, y_val).item()
    print(f'pruned: {amount*100} %, number of parameters: {params_sum*(1-amount)},  fit_before_retrained: {fitrate_before_retrained}, rms_before_retrainede: {rmse_hat_before_retrained}')

    # 枝刈り後のモデルをプロット
    plt.figure()
    plt.plot(u_val.detach().cpu().numpy(), y_val.detach().cpu().numpy())
    plt.plot(u_val.detach().cpu().numpy(), yhat_val.detach().cpu().numpy())
    plt.grid()
    plt.xlabel('u')
    plt.ylabel('y')
    plt.savefig(f'{FIGURE_FOLDER}fig_validation_before_retrained_{amount*100:.0f}.png')


    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())
    # optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.0001)

    loss_history = []
    grad_history = []
    epoch_history = []
    fit_val_history = []
    tic()   # 学習時間測定開始
    for epoch in range(EPOCHS):
        optimizer.zero_grad()

        yhat = model(u)
        loss = criterion(yhat, y)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print('epoch: ', epoch, ' loss (train): ', loss.item())

        yhat_val = model(u_val)

        loss_history.append(loss.item())
        epoch_history.append(epoch)
        grad_history.append(0)

        fit_val_history.append(fit(yhat_val, y_val))
    toc()   # 学習時間測定終了

    # 学習済みモデルを保存
    for i in range(NUM_LAYER-1):
        prune.remove(model.fc_list[i], 'weight')
        prune.remove(model.fc_list[i], 'bias')
    model_for_save = deepcopy(model)
    model_path_retrained = f'{FIGURE_FOLDER}retrained_{amount*100:.0f}.pth'
    torch.save(model_for_save.to('cpu').state_dict(), model_path_retrained)
    print('The trained model has been saved at ' + model_path_retrained)

    # パラメータ数表示
    params_pruned = 0
    for p in model.parameters():
        if p.requires_grad:
            nz_idx = torch.nonzero(torch.nn.utils.parameters_to_vector(p))
            params_pruned += len(nz_idx)
            # print(p)
    print(f'# of params: {params_pruned}')
    # sys.exit()

    # 学習ログを保存
    log_train = np.array([epoch_history, loss_history, grad_history, fit_val_history]).transpose()
    np.savetxt(f'{FIGURE_FOLDER}log_{amount*100:.0f}.csv', log_train, delimiter=',')
    print('The log has been saved at ' + FIGURE_FOLDER + 'log.csv')

    # 損失関数の変化をプロット
    plt.figure()
    plt.plot(loss_history)
    plt.yscale('log')
    plt.grid()
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.savefig(f'{FIGURE_FOLDER}fig_loss_{amount*100:.0f}.png')

    
    yhat_val = model(u_val)
    fitrate_retrained = fit(yhat_val, y_val)
    rmse_hat_retrained = rmse(yhat_val, y_val).item()
    print(f'pruned: {amount*100} %, number of parameters: {params_sum*(1-amount)},  fit: {fitrate_retrained}, rmse: {rmse_hat_retrained}')


    amount_log.append(amount)
    params_log.append(params_sum*(1-amount))
    fit_before_retrained_log.append(fitrate_before_retrained)
    rmse_before_retrained_log.append(rmse_hat_before_retrained)
    fit_retrained_log.append(fitrate_retrained)
    rmse_retrained_log.append(rmse_hat_retrained)

    # 枝刈り後のモデルをプロット
    plt.figure()
    plt.plot(u_val.detach().cpu().numpy(), y_val.detach().cpu().numpy())
    plt.plot(u_val.detach().cpu().numpy(), yhat_val.detach().cpu().numpy())
    plt.grid()
    plt.xlabel('u')
    plt.ylabel('y')
    plt.savefig(f'{FIGURE_FOLDER}fig_validation_retrained_{amount*100:.0f}.png')

    print()

# ログを保存
result = np.array([amount_log, params_log, fit_retrained_log, rmse_retrained_log, fit_before_retrained_log, rmse_before_retrained_log]).transpose()
np.savetxt(outfile_path, result, delimiter=',')

plt.show()
