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

def step(x):
  return 1.0 * (x >= 0.0)

def main(num_layer, hidden_size, date):
    # 階層的DNNとFNNを枝刈りして，検証データに対する出力を比較

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    date = f'{date}-{num_layer}-{hidden_size}'
    FIGURE_FOLDER = './figures/HierDNN/proposed/sinc/' + date + '/'
    model_path = './TrainedModels/HierDNN_cold_' + date + '.pth'    # 学習済みの階層的DNNのパス
    # model_path = './TrainedModels/HierDNN_sinc_' + date + '.pth'    # 学習済みの階層的DNNのパス
    model_path_DNN = './TrainedModels/DNN_sinc_2022-04-18-5-64.pth'      # 学習済みのFNNのパス
    # model_path_DNN = './TrainedModels/DNN_sinc_' + date + '.pth'      # 学習済みのFNNのパス

    outfile_path = FIGURE_FOLDER + 'accuracy.csv'   # 適合率の出力ファイルパス

    # 階層的DNN読み込み
    INPUT_SIZE = 1
    OUTPUT_SIZE = 1
    NUM_LAYER = num_layer
    HIDDEN_SIZE = hidden_size
    model = HierarchicalNeuralNetwork(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, output_size=OUTPUT_SIZE, num_layer=NUM_LAYER, nonlinearity='relu')
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model = try_gpu(model)

    # パラメータ数計算
    params_sum = 0
    for p in model.parameters():
        if p.requires_grad:
            params_sum  += p.numel()
    print(params_sum)

    params = []
    params.append(INPUT_SIZE*OUTPUT_SIZE + OUTPUT_SIZE)
    params.append(params[-1] + INPUT_SIZE*HIDDEN_SIZE + HIDDEN_SIZE + HIDDEN_SIZE*OUTPUT_SIZE + OUTPUT_SIZE)
    for i in range(1, NUM_LAYER-2):
        params.append(params[-1] + HIDDEN_SIZE*HIDDEN_SIZE + HIDDEN_SIZE + HIDDEN_SIZE*OUTPUT_SIZE + OUTPUT_SIZE)   # i+1層の階層的NNのパラメータ数

    params = np.array(params)
    amounts = 1 - params/params_sum
    print(params)

    # 対象システム定義
    N = 1000
    u = torch.linspace(start=-10, end=10, steps=N, device=device)
    u = u.reshape(-1, 1)
    y = torch.sin(u) / u
    # y = step(u)

    amount_log = []
    params_log = []
    fit_log = []
    rmse_log = []

    for i in range(NUM_LAYER-1):
        yhat, _intermediate_y_hat = model(u, num_used_layers=i+2)   # i層使用したモデルの出力

        # モデル出力をプロット
        plt.figure()
        plt.plot(u.detach().cpu().numpy(), y.detach().cpu().numpy(), '-k', label='True value')
        plt.plot(u.detach().cpu().numpy(), yhat.detach().cpu().numpy(), '-.r', label='Model output')
        plt.grid()
        plt.xlabel('u')
        plt.ylabel('y')
        plt.legend()
        plt.savefig(f'{FIGURE_FOLDER}pruned/fig_validation_{i}.png')

        # 精度を計算
        fitrate = fit(yhat, y)
        rmse_hat = rmse(yhat, y).item()
        print(f'pruned: {amounts[i]*100} %, number of parameters: {params[i]},  fit: {fitrate}, rmse: {rmse_hat}')

        amount_log.append(amounts[i])
        params_log.append(params[i])
        fit_log.append(fitrate)
        rmse_log.append(rmse_hat)


        # FNNを読み込んで同じ割合で枝刈り
        model_DNN = DeepNeuralNetwork(input_size=1, hidden_size=64, output_size=1, num_layer=5, nonlinearity='relu')
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
            amount=amounts[i],
        )

        yhat_DNN = model_DNN(u)


        # 枝刈り後の階層的DNNとFNNを比較
        plt.figure(tight_layout=True, figsize=[6.4, 3.5])
        plt.rcParams['pdf.fonttype'] = 42
        plt.rcParams['ps.fonttype'] = 42
        plt.rcParams["font.family"] = "Times New Roman" 
        plt.rcParams["mathtext.fontset"] = "cm"
        plt.rcParams["font.size"] = 15
        plt.plot(u.detach().cpu().numpy(), y.detach().cpu().numpy(), '-k', label='True value')
        # plt.plot(u.detach().cpu().numpy(), yhat.detach().cpu().numpy(), '-.r', label=f'Proposed model ($l$={i+1})')
        # plt.plot(u.detach().cpu().numpy(), yhat_DNN.detach().cpu().numpy(), '--b', label=f'Pruned FNN ({amounts[i]*100:.02f} % pruned)')
        plt.plot(u.detach().cpu().numpy(), yhat.detach().cpu().numpy(), '-.r', label=f'Proposed model')
        plt.plot(u.detach().cpu().numpy(), yhat_DNN.detach().cpu().numpy(), '--b', label=f'Pruned FNN')
        plt.plot(u.detach().cpu().numpy(), 0.166*torch.ones_like(u).detach().cpu().numpy(), ':g', linewidth = 3, label='BLA')
        plt.grid()
        plt.xlabel('$u$')
        plt.ylabel('$y$')
        # plt.legend(loc='upper right')
        plt.ylim(-0.4, 1.1)
        plt.savefig(f'{FIGURE_FOLDER}pruned/fig_comparison_BLA_{i}.pdf')
        plt.savefig(f'{FIGURE_FOLDER}pruned/fig_comparison_BLA_{i}.png')
        # plt.close()

    # ログ保存
    result = np.array([amount_log, params_log, fit_log, rmse_log]).transpose()
    np.savetxt(outfile_path, result, delimiter=',')

    # plt.show()



if __name__ == '__main__':
    # main(num_layer=5, hidden_size=64, date='2023-01-24')

    for trial in range(10):
        main(num_layer=5, hidden_size=64, date=f'2023-01-24-{trial}')
