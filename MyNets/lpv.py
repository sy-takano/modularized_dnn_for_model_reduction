#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
from MyNets.dnn import DeepNeuralNetwork

class LPVNARX(nn.Module):
    # パラメータがスケジューリングパラメータに依存するARXモデル
    # NARXモデルによるLPVモデルの表現の一つ？
    def __init__(self, input_size, output_size, hidden_size, num_layer, nonlinearity = 'relu'):
        super().__init__()
        self.scheduling_variable_size = 1

        self.scheduling_estimator = DeepNeuralNetwork(input_size=input_size, output_size=self.scheduling_variable_size, hidden_size=hidden_size, num_layer=num_layer, nonlinearity=nonlinearity)
        self.compressor = nn.Sigmoid()

        self.theta_estimator = nn.Linear(self.scheduling_variable_size, input_size)
    
    def forward(self, phi):
        scheduling_parameter = self.compressor(self.scheduling_estimator(phi))
        theta = self.theta_estimator(scheduling_parameter)
        y = torch.sum(theta * phi, dim=1, keepdim=True)
        return y