import torch

def standardize(x):
    # xの各列に対して，それぞれ平均0, 標準偏差1に標準化
    x_dim = x.size()[1]
    for i in range(x_dim):
        std, mu = torch.std_mean(x[:, i])
        x[:, i] = (x[:, i] - mu) /std
    return x