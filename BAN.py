import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm


class BANLayer(nn.Module):
    def __init__(self, v_dim, q_dim, h_dim, h_out, act='ReLU', dropout=0.2, k=3):
        super(BANLayer, self).__init__()

        self.c = 32
        self.k = k # 3
        self.v_dim = v_dim # 128
        self.q_dim = q_dim # 128
        self.h_dim = h_dim # 256
        self.h_out = h_out # 2

        # v_dim, q_dim = 128, h_dim = 256, k = 3
        self.v_net = FCNet([v_dim, h_dim * self.k], act=act, dropout=dropout) # [128, 768]
        self.q_net = FCNet([q_dim, h_dim * self.k], act=act, dropout=dropout) # [128, 768]
        # self.dropout = nn.Dropout(dropout[1])
        if 1 < k:
            self.p_net = nn.AvgPool1d(self.k, stride=self.k)

        if h_out <= self.c:
            self.h_mat = nn.Parameter(torch.Tensor(1, h_out, 1, h_dim * self.k).normal_())
            self.h_bias = nn.Parameter(torch.Tensor(1, h_out, 1, 1).normal_())
        else:
            self.h_net = weight_norm(nn.Linear(h_dim * self.k, h_out), dim=None)

        self.bn = nn.BatchNorm1d(h_dim)

    def attention_pooling(self, v, q, att_map):
        fusion_logits = torch.einsum('bvk,bvq,bqk->bk', (v, att_map, q))
        # batch, 768
        if 1 < self.k:
            fusion_logits = fusion_logits.unsqueeze(1)  # b x 1 x d # b, 1, 768
            fusion_logits = self.p_net(fusion_logits).squeeze(1) * self.k  # sum-pooling #
        #
        return fusion_logits

    def forward(self, v, q, softmax=False):
        # print(v.shape, q.shape)
        # print('====================================')
        v_num = v.size(1) # v.size [batch_size(8), drug_representation = (290, hidden_dimension(128))]
        q_num = q.size(1) # q.size [batch_size(8), protein_representation = (1185, hidden_dimension(128))]
        if self.h_out <= self.c:
            v_ = self.v_net(v) # 8, 290, 768
            q_ = self.q_net(q) # 8, 32, 768
            att_maps = torch.einsum('xhyk,bvk,bqk->bhvq', (self.h_mat, v_, q_)) + self.h_bias
            # print(self.h_mat.size()) # 1 2 1 768
            # print(self.h_bias.size()) # 1 2 1 1
            # print(att_maps.shape) # 8 2 290 32

        else:
            v_ = self.v_net(v).transpose(1, 2).unsqueeze(3)
            q_ = self.q_net(q).transpose(1, 2).unsqueeze(2)
            d_ = torch.matmul(v_, q_)  # b x h_dim x v x q
            att_maps = self.h_net(d_.transpose(1, 2).transpose(2, 3))  # b x v x q x h_out
            att_maps = att_maps.transpose(2, 3).transpose(1, 2)  # b x h_out x v x q
        if softmax:
            p = nn.functional.softmax(att_maps.view(-1, self.h_out, v_num * q_num), 2)
            att_maps = p.view(-1, self.h_out, v_num, q_num)
        logits = self.attention_pooling(v_, q_, att_maps[:, 0, :, :])
        for i in range(1, self.h_out):
            logits_i = self.attention_pooling(v_, q_, att_maps[:, i, :, :])
            logits += logits_i
        logits = self.bn(logits)
        return logits


class FCNet(nn.Module):
    """Simple class for non-linear fully connect network
    Modified from https://github.com/jnhwkim/ban-vqa/blob/master/fc.py
    """

    def __init__(self, dims, act='ReLU', dropout=0):
        super(FCNet, self).__init__()

        # dims = [128, 768]
        layers = []
        for i in range(len(dims) - 2):
            in_dim = dims[i]
            out_dim = dims[i + 1]
            if 0 < dropout:
                layers.append(nn.Dropout(dropout))
            layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
            if '' != act:
                layers.append(getattr(nn, act)())
        if 0 < dropout:
            layers.append(nn.Dropout(dropout))
        layers.append(weight_norm(nn.Linear(dims[-2], dims[-1]), dim=None))
        if '' != act:
            layers.append(getattr(nn, act)())

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)