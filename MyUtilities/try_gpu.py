#!/usr/bin/env python
# conding: utf-8

import torch

def try_gpu(e):
    # cudaが使えるならeをgpuに移す
    if torch.cuda.is_available():
        return e.cuda()
    return e