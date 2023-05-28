#!/usr/bin/env python
# coding: utf-8

import torch.nn as nn

class DeepNeuralNetwork(nn.Module):
    # 順伝播型ニューラルネットワーク
    def __init__(self, input_size, output_size, hidden_size, num_layer, nonlinearity='relu', bias=True):
        # num_layer: 入出力層も含めた層数 例；入力→（アフィン変換）→活性化関数（隠れ層）→（アフィン変換）→出力は３層
        if num_layer < 3:
            raise NotImplementedError()

        super().__init__()

        self.num_layer = num_layer

        if nonlinearity == 'relu':
            self.activation = nn.ReLU()
        elif nonlinearity == 'tanh':
            self.activation = nn.Tanh()
        elif nonlinearity == 'sigmoid':
            self.activation = nn.Sigmoid()

        nodes_each_layer = [hidden_size]*(num_layer-2)
        nodes_each_layer.insert(0, input_size)
        nodes_each_layer.append(output_size)

        self.fc_list = nn.ModuleList([
            nn.Linear(nodes_each_layer[i], nodes_each_layer[i+1], bias=bias) for i in range(num_layer)[:-1]])   # 全結合層
    
    def forward(self, x):
        x = self.fc_list[0](x)
        x = self.activation(x)

        for i in range(self.num_layer)[1:-2]:
            x = self.fc_list[i](x)
            x = self.activation(x)

        x = self.fc_list[-1](x)
        return x
