#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn

def try_gpu(e):
    if torch.cuda.is_available():
        return e.cuda()
    return e

class HierarchicalNeuralNetwork(nn.Module):
    # 階層的DNN
    def __init__(self, input_size, hidden_size, output_size, num_layer, nonlinearity='relu', send_gpu_state=True):
        if num_layer < 3:
            raise NotImplementedError()

        super().__init__()

        if nonlinearity == 'relu':
            self.activation = nn.ReLU()
        elif nonlinearity == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif nonlinearity == 'elu':
            self.activation = nn.ELU()

        self.output_size = output_size
        self.num_layer = num_layer

        nodes_each_layer_in = [hidden_size]*(num_layer-2)
        nodes_each_layer_in.insert(0, input_size)

        nodes_each_layer_out = [output_size + hidden_size]*(num_layer-2)
        nodes_each_layer_out.append(output_size)

        self.fc_list = nn.ModuleList([
            nn.Linear(nodes_each_layer_in[i], nodes_each_layer_out[i]) for i in range(num_layer - 1)]) # 全結合層

    def forward(self, x, num_used_layers=None):
        if num_used_layers == None:
            num_used_layers = self.num_layer
        if num_used_layers < 2:
            raise NotImplementedError()

        intermediate_output = []
        
        h = self.fc_list[0](x)
        output = h[:, :self.output_size]
        intermediate_output.append(h[:, :self.output_size])

        for i in range(1, num_used_layers-1):
            h = self.activation(h[:, self.output_size:])
            h = self.fc_list[i](h)
            output = output + h[:, :self.output_size]
            intermediate_output.append(h[:, :self.output_size])

        return output.reshape(-1, self.output_size), intermediate_output



