#!/usr/bin/env python
# conding: utf-8

import torch
import numpy as np
import matplotlib.pyplot as plt

def get_grad(parameters):
    # パラメータの勾配の要素の絶対値の平均を取得
    num_params = 0
    average_grad = torch.tensor(0.0)
    for param in parameters:
        if param.requires_grad:
            average_grad = average_grad + torch.sum(torch.abs(param.grad))
            num_params = num_params + param.numel()
    
    average_grad = average_grad / num_params
    return average_grad

def get_grad_histogram(parameters):
    # パラメータの勾配のヒストグラムを出力
    grad = np.ndarray([])
    for param in parameters:
        grad = np.append(grad, param.grad.detach().cpu().numpy())
    
    grad = grad[grad<1]
    grad_hist, bins = np.histogram(grad, bins=100)

    return grad_hist, bins

def plot_histogram(parameters):
    # パラメータの勾配のヒストグラムをプロット
    num_tensors = 0
    grad = np.ndarray([])
    for param in parameters:
        grad = np.append(grad, param.grad.detach().cpu().numpy())
    
    # grad = grad[grad<10]
    
    plt.hist(grad, bins=100)
