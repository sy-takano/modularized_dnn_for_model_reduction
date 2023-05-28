#!/usr/bin/env python
# conding: utf-8

import torch

def fit(y_hat, y_data):  # 適合率（%）を計算
    fitvalue = (1-torch.norm(y_hat-y_data)/torch.norm(y_data-torch.mean(y_data)))*100
    return fitvalue.item()
