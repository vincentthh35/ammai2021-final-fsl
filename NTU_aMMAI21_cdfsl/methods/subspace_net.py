import backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from methods.meta_template import MetaTemplate

import utils

"""
    this class is mainly based on deep subspace
    network implementation:
    https://github.com/chrysts/dsn_fewshot/
"""

class SubspaceNet(MetaTemplate):
    def __init__(self, model_func, n_way, n_support, n_dim=4, lamb=0.03):
        super(SubspaceNet, self).__init__( model_func, n_way, n_support )
        self.loss_fn = nn.CrossEntropyLoss()
        self.n_dim = n_dim
        self.lamb = lamb # for scaling discriminative loss

    def create_subspace(self, all_feature):
        all_basis = []
        means = []
        l1_reg = 0
        for i in range(self.n_way):
            n_sample = self.n_support
            in_class_feature = all_feature[i]
            meann = torch.mean(in_class_feature, dim=0)
            means.append(meann)

            in_class_feature = in_class_feature - meann.unsqueeze(0).repeat(n_sample, 1)
            in_class_feature = torch.transpose(in_class_feature, 0, 1)
            uu, _, _ = torch.svd(in_class_feature.double(), some=False)
            uu = uu.float()
            all_basis.append(uu[:, :self.n_dim])
            l1_reg = l1_reg + torch.norm(uu, p=1)
        # print(l1_reg)

        all_basis = torch.stack(all_basis, dim=0)
        means = torch.stack(means)

        # ?
        if len(all_basis.size()) < 3:
            # unsqueeze?
            all_basis = all_basis.unsqueeze(-1)

        return all_basis, means

    def projection(self, all_feature, all_basis, means):
        support_size = all_feature.shape[0]
        batch_size = all_feature.shape[1]
        class_size = all_basis.shape[0]
        feature_size = all_feature.shape[-1]
        eps = 1e-12

        # all_feature = all_feature.permute(1, 2, 0)

        logits = []

        discriminative_loss = 0.0

        for i in range(class_size):
            h_plane = all_basis[i].unsqueeze(0).repeat(batch_size, 1, 1)
            expanded_feature = (all_feature - means[i].expand_as(all_feature)).permute(1, 2, 0)
            # P_c P_c^T f(q)
            # print(all_feature.shape)
            # print(h_plane.shape, (torch.transpose(h_plane, 1, 2)).shape, expanded_feature.shape)
            projection_i = torch.bmm(h_plane, torch.bmm(torch.transpose(h_plane, 1, 2), expanded_feature)).transpose(1, 2)
            # print(torch.squeeze(projection_i).shape, means[i].unsqueeze(0).repeat(batch_size, support_size, 1).shape)
            projection_i = projection_i + means[i].unsqueeze(0).repeat(batch_size, support_size, 1)
            projection_dist = all_feature - projection_i.transpose(0, 1)
            # print(projection_dist.shape)

            # sqrt distance (slower)
            distance = -torch.sqrt(torch.sum(projection_dist * projection_dist, dim=-1) + eps)
            # squared distance (faster)
            # distance = -torch.sum(projection_dist * projection_dist, dim=-1)
            # print(distance.shape)
            logits.append(distance.view(support_size * batch_size))

            for j in range(class_size):
                if i != j:
                    temp = torch.mm(torch.transpose(all_basis[i], 0, 1), all_basis[j])
                    discriminative_loss = discriminative_loss + torch.sum(temp * temp)


        return torch.stack(logits, dim=1), discriminative_loss

    def correct(self, x):
        scores, _  = self.set_forward(x)
        y_query = np.repeat(range( self.n_way ), self.n_query )

        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()
        top1_correct = np.sum(topk_ind[:,0] == y_query)
        return float(top1_correct), len(y_query)

    def set_forward(self, x, is_feature=False):
        z_support, z_query  = self.parse_feature(x, is_feature)
        z_support = z_support.contiguous()
        z_query = z_query.contiguous()

        all_basis, means = self.create_subspace(z_support)
        # calculate distance
        logits, discriminative_loss = self.projection(z_query, all_basis, means)

        # calculate distance
        # dists = ?
        # scores = -dists
        return logits, discriminative_loss

    def set_forward_loss(self, x):
        y_query = torch.from_numpy(np.repeat(range( self.n_way ), self.n_query ))
        y_query = Variable(y_query.to(self.device))

        scores, discriminative_loss = self.set_forward(x)
        # print(scores.shape, y_query.shape)
        loss = self.loss_fn(scores, y_query) + self.lamb * discriminative_loss

        return loss
