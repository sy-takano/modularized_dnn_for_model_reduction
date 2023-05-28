#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn

# pytorchにおけるRNN実装の習作

class MyRNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        # self.rnncell = MyRNNCell(input_size, hidden_size)
        self.rnncell = torch.nn.RNNCell(input_size, hidden_size)
    def forward(self, x):   # x: 入力 shape: (seq_len, num_batch, input_size)
        seq_len = len(x[:, 0, 0])
        batch_size = len(x[0, :, 0])

        hidden = torch.zeros(batch_size, self.hidden_size)  # 初期隠れ状態 shape: (num_layers * num_directions, batch, hidden_size)
        output = torch.Tensor()

        for index in range(seq_len):
            hidden = self.rnncell(x[index, :, :], hidden)
            output = torch.cat((output, hidden))
        output = output.reshape(seq_len, -1, self.hidden_size)
        return output, hidden


class MyRNNCell(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.fc = torch.nn.Linear(input_size + hidden_size, hidden_size)
        self.activation_function = torch.nn.Tanh()
    def forward(self, x, hidden): # x: 入力 shape: (num_batch, input_size)， hidden: 隠れ状態 shape: (num_batch, hidden_size)
        num_batch = len(x[:, 0])
        output = torch.Tensor()
        for index in range(num_batch):
            x_temp = x[index, :].reshape(1, -1)
            h_temp = hidden[index, :].reshape(1, -1)
            temp = torch.cat((x_temp, h_temp), 1)
            temp = self.fc(temp)
            temp = self.activation_function(temp)
            output = torch.cat((output, temp), 0)
        return output


class MyNaiveRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.hidden_layer = nn.Linear(hidden_size + input_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.activation = nn.Tanh()
    
    def forward(self, input_seq, hidden_state):
        input_seq = input_seq.reshape(-1, self.input_size)
        seq_size = len(input_seq)

        output = torch.zeros(seq_size, self.output_size)

        for time in range(seq_size):
            x = torch.cat([input_seq[time], hidden_state])
            hidden_state = self.hidden_layer(x)
            hidden_state = self.activation(hidden_state)
            output[time] = self.output_layer(hidden_state)
        
        return output


class Predictor(nn.Module):
    def __init__(self, input_size, state_size, output_size, hidden_size):
        super().__init__()

        self.input_size = input_size
        self.state_size = state_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        self.fc1 = nn.Linear(state_size + input_size, hidden_size)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, state_size)
    
    def forward(self, input_seq, initial_state):
        state = initial_state







