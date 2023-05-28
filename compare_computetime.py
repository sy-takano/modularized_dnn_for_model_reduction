#!/usr/bin/env python
# coding: utf-8

import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import matplotlib.pyplot as plt

from MyNets.hierarchical_dnn import HierarchicalNeuralNetwork
from MyNets.dnn import DeepNeuralNetwork

from MyUtilities.try_gpu import try_gpu
from MyUtilities.measure_elapsed_time import tic, toc
from MyUtilities.rmse import rmse
from MyUtilities.fit import fit


# 学習済みモデルの推論にかかる計算時間を計測


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


date = '2021-12-16'
FIGURE_FOLDER = './figures/HierDNN/' + date + '/'
model_path = './TrainedModels/HierDNN_sinc_' + date + '.pth'
model_path_DNN = './TrainedModels/DNN_sinc_' + date + '.pth'

outfile_path = FIGURE_FOLDER + 'accuracy.csv'


NUM_LAYER = 5
model = HierarchicalNeuralNetwork(input_size=1, hidden_size=64, output_size=1, num_layer=NUM_LAYER, nonlinearity='relu')
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model = try_gpu(model)


params = 0
for p in model.parameters():
    if p.requires_grad:
        params += p.numel()
print(params)

params = np.array([2., 195., 4420., 8645.])
amounts = 1 - params/8645



N = 1000
u = torch.linspace(start=-10, end=10, steps=N, device=device)
u = u.reshape(-1, 1)
y = torch.sin(u) / u


amount_log = []
params_log = []
fit_log = []
rmse_log = []
computation_time_log = []

params_DNN_log = []
fit_DNN_log = []
computation_time_DNN_log = []

trials = 10000

for i in range(NUM_LAYER):
    # 階層的NNの平均計算時間を求める
    tic()
    for jj in range(trials):
        yhat, _intermediate_y_hat = model(u, num_used_layers=i+1)

    elapsed_time = toc()
    computation_time_per_once = elapsed_time / trials


    fitrate = fit(yhat, y)
    rmse_hat = rmse(yhat, y).item()
    print(f'pruned: {amounts[i]*100} %, number of parameters: {params[i]},  fit: {fitrate}, rmse: {rmse_hat}, computation time: {computation_time_per_once}')

    amount_log.append(amounts[i])
    params_log.append(params[i])
    fit_log.append(fitrate)
    rmse_log.append(rmse_hat)
    computation_time_log.append(computation_time_per_once)


    # FNNの平均計算時間を求める
    model_DNN = DeepNeuralNetwork(input_size=1, hidden_size=64, output_size=1, num_layer=NUM_LAYER+1, nonlinearity='relu')
    model_DNN.load_state_dict(torch.load(model_path_DNN, map_location=torch.device('cpu')))
    model_DNN = try_gpu(model_DNN)

    parameters_to_prune = (
        (model_DNN.fc_list[0], 'weight'),
        (model_DNN.fc_list[0], 'bias'),
        (model_DNN.fc_list[1], 'weight'),
        (model_DNN.fc_list[1], 'bias'),
        (model_DNN.fc_list[2], 'weight'),
        (model_DNN.fc_list[2], 'bias'),
        (model_DNN.fc_list[3], 'weight'),
        (model_DNN.fc_list[3], 'bias')
    )

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amounts[i]
    )

    tic()
    for jj in range(trials):
        yhat_DNN = model_DNN(u)

    elapsed_time_DNN = toc()

    fitrate_DNN = fit(yhat_DNN, y)
    computation_time_per_once_DNN = elapsed_time_DNN / trials
    
    print(f'fit (pruned DNN): {fitrate_DNN}, computation time (pruned DNN): {computation_time_per_once_DNN}')

    params_DNN_log.append(8440*(1 - amounts[i]))
    fit_DNN_log.append(fitrate_DNN)
    computation_time_DNN_log.append(computation_time_per_once_DNN)

    print()



result = np.array([amount_log, params_log, fit_log, rmse_log]).transpose()
np.savetxt(outfile_path, result, delimiter=',')


plt.figure()
plt.plot(params_log, [x*(10**3) for x in computation_time_log], '-.r',  marker='o', label=f'Proposed model')
plt.plot(params_DNN_log, [x*(10**3) for x in computation_time_DNN_log], '--b', marker='o', label=f'Pruned DNN')
plt.grid()
plt.xlabel('Number of parameters')
plt.ylabel('Computation time [ms]')
plt.legend()
plt.savefig(FIGURE_FOLDER + 'fig_params_vs_time.png')


plt.figure()
plt.plot([x*(10**3) for x in computation_time_log], fit_log, '-.r',  marker='o', label=f'Proposed model')
plt.plot([x*(10**3) for x in computation_time_DNN_log], fit_DNN_log, '--b', marker='o', label=f'Pruned DNN')
plt.grid()
plt.xlabel('Computation time [ms]')
plt.ylabel('Fit rate [%]')
plt.legend()
plt.savefig(FIGURE_FOLDER + 'fig_time_vs_accuracy.png')

plt.figure()
plt.plot(params_log, fit_log, '-.r',  marker='o', label=f'Proposed model')
plt.plot(params_DNN_log, fit_DNN_log, '--b', marker='o', label=f'Pruned DNN')
plt.grid()
plt.xlabel('Number of parameters')
plt.ylabel('Fit rate [%]')
plt.legend()
plt.savefig(FIGURE_FOLDER + 'fig_params_vs_accuracy.png')

plt.show()
