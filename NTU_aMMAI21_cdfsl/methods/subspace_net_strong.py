from methods.subspace_net_plus import SubspaceNetPlus
import torch.nn as nn
import torch

class SubspaceNetStrong(SubspaceNetPlus):
    def __init__(self, model_func, n_way, n_support):
        super(SubspaceNetStrong, self).__init__(model_func, n_way, n_support)
        self.L = nn.Linear(512, 512)

    def forward(self, x):
        return self.L(super(SubspaceNetStrong, self).forward(x))
