#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from MyUtilities.get_grad import get_grad, plot_histogram
from MyUtilities.fit import fit


def train_signal_regularization(model, u, y, num_layer, regularization_parameter, path_histogram, u_val=0, y_val=0, epochs=10000, device='cpu'):
    # 階層的DNNの学習
    criterion = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters())

    loss_history = []
    loss_history_termwise = np.zeros((epochs, num_layer-1))
    grad_history = []
    epoch_history = []

    fit_val_history = []

    for epoch in range(epochs):
        optimizer.zero_grad()

        loss = torch.tensor([0.0], device=device, requires_grad=True)

        yhat, intermediate_y_hat = model(u, num_used_layers=num_layer)
        o = torch.zeros_like(intermediate_y_hat[0])

        for i in range(num_layer-1):
            o = o + intermediate_y_hat[i]
            loss = loss + regularization_parameter[i] * criterion(o.reshape(-1, 1), y)

            # yhat, _intermediate_y_hat = model(u, num_used_layers=i+2)
            # loss = loss + regularization_parameter[i] * criterion(yhat, y)

        loss.backward()
        optimizer.step()

        try:
            grad = get_grad(model.parameters())
        except:
            grad = torch.tensor([0])    # 勾配を取得できない場合

        if epoch % 100 == 0:
            print('epoch: ', epoch, ' loss: ', loss.item(), 'grad: ', grad.item())
        
        # if epoch % 1000 == 0:
        #     plot_histogram(model.parameters())
        #     plt.grid()
        #     plt.savefig(f'{path_histogram}/fig_hist_{epoch}.png')
        #     plt.close()

        yhat_val, _intermediate_y_hat = model(u_val)

        loss_history.append(loss.item())
        grad_history.append(grad.item())
        epoch_history.append(epoch)

        fit_val_history.append(fit(yhat_val, y_val))

        for i in range(num_layer-1):
            yhat_termwise, _intermediate_y_hat = model(u, num_used_layers=i+2)
            loss_termwise = criterion(yhat_termwise, y)
            loss_history_termwise[epoch, i] = loss_termwise.item()

    
    return loss_history, loss_history_termwise, epoch_history, grad_history, fit_val_history

def retrain_signal_regularization(model, u, y, num_layer, path_histogram, num_used_layer, u_val=0, y_val=0, epochs=10000, device='cpu'):
    # 階層的DNNの再学習
    
    criterion = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters())

    loss_history = []
    grad_history = []
    epoch_history = []

    fit_val_history = []

    for epoch in range(epochs):
        optimizer.zero_grad()
        
        yhat, _intermediate_y_hat = model(u, num_used_layers=num_used_layer)
        loss = criterion(yhat, y)

        loss.backward()
        optimizer.step()

        # grad = get_grad(model.parameters())

        if epoch % 100 == 0:
            # print('epoch: ', epoch, ' loss: ', loss.item(), 'grad: ', grad.item())
            print('epoch: ', epoch, ' loss: ', loss.item())

        
        # if epoch % 1000 == 0:
        #     plot_histogram(model.parameters())
        #     plt.grid()
        #     plt.savefig(f'{path_histogram}/fig_hist_{epoch}.png')
        #     plt.close()

        yhat_val, _intermediate_y_hat = model(u_val)

        loss_history.append(loss.item())
        # grad_history.append(grad.item())
        grad_history.append(0)
        epoch_history.append(epoch)

        fit_val_history.append(fit(yhat_val, y_val))
    
    return loss_history, epoch_history, grad_history, fit_val_history


def train_stochastic_costfunc(model, u, y, num_layer, epochs=10000, device='cpu'):
    # train_signal_regularizationを確率的に近似
    # 各イタレーションで層数をランダムに選択
    
    criterion = torch.nn.MSELoss()
    criterion_intermediate = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters())

    loss_history = []
    term_number_history = []
    
    regularization_parameter = [1.0, 1.0, 1.0, 1.0]
    probability = [gamma / sum(regularization_parameter) for gamma in regularization_parameter]

    for epoch in range(epochs):
        optimizer.zero_grad()

        term_number = np.random.choice(num_layer, p=probability) + 1
        yhat, _intermediate_y_hat = model(u, num_used_layers=term_number)

        loss = criterion(yhat, y)

        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print('epoch: ', epoch, ' loss: ', loss.item())

        loss_history.append(loss.item())
        term_number_history.append(term_number)
    
    return loss_history, term_number_history
    