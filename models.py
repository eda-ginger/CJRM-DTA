########################################################################################################################
########## Sources
########################################################################################################################

# https://github.com/hkmztrk/DeepDTA
# https://github.com/thinng/GraphDTA
# https://github.com/peizhenbai/DrugBAN
# https://github.com/595693085/DGraphDTA/tree/master
# https://github.com/JK-Liu7/AttentionMGT-DTA/tree/main
# https://github.com/zhaoqichang/AttentionDTA_TCBB/tree/main

########################################################################################################################
########## Import
########################################################################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch.nn.utils.weight_norm import weight_norm
from torch_geometric.nn import GINConv, global_add_pool as gap
from torch_geometric.nn import GCNConv, global_mean_pool as gep

from BAN import BANLayer
from MHA import CrossAttention, CoAttention

########################################################################################################################
########## Models
########################################################################################################################

# d1 (seq) & d2 (seq) - DeepDTA
class SnS(torch.nn.Module):
    def __init__(self, n_output=1, num_features_xd=64, num_features_xt=25,
                 n_filters=32, embed_dim=128, output_dim=128, joint='concat'):

        super(SnS, self).__init__()

        self.joint = joint
        self.relu = nn.ReLU()
        self.n_filters = n_filters

        jdim = 256
        kd_size = 4

        if self.joint == 'bi':
            self.jc = Bilinear_Joint(output_dim, jdim)

        elif 'att' in self.joint:
            kd_size = 8
            if self.joint == 'bi_att':
                self.jc = weight_norm(BANLayer(v_dim=107, q_dim=107, h_dim=jdim, h_out=2), name='h_mat', dim=None)
            elif self.joint == 'cross_att':
                self.jc = CrossAttention(n_filters * 3, n_filters * 3, embed_dim=107, out_dim=jdim, num_heads=8)
            elif self.joint == 'co_att':
                self.jc = CoAttention(n_filters * 3, n_filters * 3, embed_dim=107, out_dim=jdim, num_heads=8)
            else:
                raise Exception('wrong att type')
        else:
            self.jc = Simple_Joint(self.joint)
            if self.joint != 'concat':
                jdim = 128

        # 1D convolution on smiles sequence
        self.embedding_xd = nn.Embedding(num_features_xd + 1, embed_dim) # batch, 100, 128
        self.conv_xd_1 = nn.Conv1d(in_channels=100, out_channels=n_filters, kernel_size=kd_size) # batch, 32, 121
        self.conv_xd_2 = nn.Conv1d(in_channels=n_filters, out_channels=n_filters * 2, kernel_size=kd_size) # batch, 64, 114
        self.conv_xd_3 = nn.Conv1d(in_channels=n_filters * 2, out_channels=n_filters * 3, kernel_size=kd_size) # batch, 96, 107

        # 1D convolution on protein sequence
        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim) # (1000, 1) -> (1000, 128)
        self.conv_xt_1 = nn.Conv1d(in_channels=1000, out_channels=n_filters, kernel_size=8) # batch, 32, 121
        self.conv_xt_2 = nn.Conv1d(in_channels=n_filters, out_channels=n_filters * 2, kernel_size=8) # batch, 64, 114
        self.conv_xt_3 = nn.Conv1d(in_channels=n_filters * 2, out_channels=n_filters * 3, kernel_size=8) # batch, 96, 107

        if not 'att' in self.joint:
            self.pool_xd = nn.AdaptiveMaxPool1d(1)  # batch, 96, 1
            self.fc1_xd = nn.Linear(n_filters * 3, output_dim)  # batch, 128

            self.pool_xt = nn.AdaptiveMaxPool1d(1)  # batch, 96, 1
            self.fc1_xt = nn.Linear(n_filters * 3, output_dim)  # batch, 128

        # dense
        self.classifier = nn.Sequential(
            nn.Linear(jdim, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, n_output),
        )

    def forward(self, data):
        drug, target, y = data
        xd, xt = drug.x, target.x

        # drug
        embedded_xd = self.embedding_xd(xd)
        conv_xd = self.relu(self.conv_xd_1(embedded_xd))
        conv_xd = self.relu(self.conv_xd_2(conv_xd))
        conv_xd = self.relu(self.conv_xd_3(conv_xd))

        # protein
        embedded_xt = self.embedding_xt(xt)
        conv_xt = self.relu(self.conv_xt_1(embedded_xt))
        conv_xt = self.relu(self.conv_xt_2(conv_xt))
        conv_xt = self.relu(self.conv_xt_3(conv_xt))

        # joint
        if 'att' in self.joint:
            xj = self.jc(conv_xd, conv_xt)
        else:
            # flatten
            xd = self.pool_xd(conv_xd)
            xd = self.fc1_xd(xd.view(-1, self.n_filters * 3)) # batch, 128
            xt = self.pool_xt(conv_xt)
            xt = self.fc1_xt(xt.view(-1, self.n_filters * 3)) # batch, 128

            xj = self.jc(xd, xt)

        # dense
        out = self.classifier(xj).squeeze(1)
        return out, y


# d1 (graph) & d2 (seq) - GraphDTA
class GnS(torch.nn.Module):
    def __init__(self, n_output=1, num_features_xd=55, num_features_xt=25, n_filters=32, embed_dim=128,
                 output_dim=128, dropout=0.2, joint='concat'):

        super(GnS, self).__init__()

        dim = 32
        jdim = 256
        self.joint = joint
        if self.joint in ['add', 'multiple']:
            jdim = 128

        # GIN layers (drug)
        nn1_xd = Sequential(Linear(num_features_xd, dim), ReLU(), Linear(dim, dim))
        self.conv1_xd = GINConv(nn1_xd)
        self.bn1_xd = torch.nn.BatchNorm1d(dim)

        nn2_xd = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv2_xd = GINConv(nn2_xd)
        self.bn2_xd = torch.nn.BatchNorm1d(dim)

        nn3_xd = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv3_xd = GINConv(nn3_xd)
        self.bn3_xd = torch.nn.BatchNorm1d(dim)

        nn4_xd = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv4_xd = GINConv(nn4_xd)
        self.bn4_xd = torch.nn.BatchNorm1d(dim)

        # 1D convolution on protein sequence
        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
        self.conv_xt_1 = nn.Conv1d(in_channels=1000, out_channels=n_filters, kernel_size=8)  # batch, 32, 121

        if 'att' in self.joint:
            nn5_xd = Sequential(Linear(dim, dim), ReLU(), Linear(dim, 121))
            self.conv5_xd = GINConv(nn5_xd) # batch, node, 121
            self.bn5_xd = torch.nn.BatchNorm1d(121)

            if self.joint == 'bi_att':
                self.jc = weight_norm(BANLayer(v_dim=121, q_dim=121, h_dim=jdim, h_out=2), name='h_mat', dim=None)
            elif self.joint == 'cross_att':
                self.jc = CrossAttention(290, n_filters, embed_dim=121, out_dim=jdim, num_heads=8)
            elif self.joint == 'co_att':
                self.jc = CoAttention(290, n_filters, embed_dim=121, out_dim=jdim, num_heads=8)
            else:
                raise Exception('wrong att type')
        else:
            nn5_xd = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
            self.conv5_xd = GINConv(nn5_xd)
            self.bn5_xd = torch.nn.BatchNorm1d(dim)

            self.fc1_xd = Linear(dim, output_dim)
            self.fc1_xt = nn.Linear(32 * 121, output_dim)

            if self.joint in ['concat', 'add', 'multiple']:
                self.jc = Simple_Joint(self.joint)
            elif self.joint == 'bi':
                self.jc = Bilinear_Joint(output_dim, jdim)
            else:
                raise Exception(f'{self.joint} method not supported!!!')

        # dense
        self.classifier = nn.Sequential(
            nn.Linear(jdim, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_output),
        )

    def forward(self, data):
        drug, target, y = data
        xd, xd_ei, xd_batch = drug.x, drug.edge_index, drug.batch
        xt = target.x

        # drug
        xd = F.relu(self.conv1_xd(xd, xd_ei))
        xd = self.bn1_xd(xd)
        xd = F.relu(self.conv2_xd(xd, xd_ei))
        xd = self.bn2_xd(xd)
        xd = F.relu(self.conv3_xd(xd, xd_ei))
        xd = self.bn3_xd(xd)
        xd = F.relu(self.conv4_xd(xd, xd_ei))
        xd = self.bn4_xd(xd)
        xd = F.relu(self.conv5_xd(xd, xd_ei))
        xd = self.bn5_xd(xd)

        embedded_xt = self.embedding_xt(xt)
        conv_xt = self.conv_xt_1(embedded_xt)

        # joint
        if 'att' in self.joint:
            xd = xd.view(len(y), 290, 121)
            xj = self.jc(xd, conv_xt)

        else:
            # flatten
            xd = gap(xd, xd_batch)
            xd = F.relu(self.fc1_xd(xd))
            xd = F.dropout(xd, p=0.2, training=self.training)
            xt = self.fc1_xt(conv_xt.view(-1, 32 * 121))
            xj = self.jc(xd, xt)

        # dense
        out = self.classifier(xj).squeeze(1)
        return out, y


# # d1 (seq) & d2 (graph) - Custom
# class SnG(torch.nn.Module):
#     def __init__(self, n_output=1, num_features_xd=64, num_features_xt=41,
#                  n_filters=32, embed_dim=128, output_dim=128, dropout=0.2, joint='concat'):
#
#         super(SnG, self).__init__()
#
#         dim = 32
#         self.joint = joint
#
#         # 1D convolution on smiles sequence
#         self.embedding_xd = nn.Embedding(num_features_xd + 1, embed_dim)
#         self.conv_xd_1 = nn.Conv1d(in_channels=100, out_channels=n_filters, kernel_size=8)
#         self.fc1_xd = nn.Linear(32 * 121, output_dim)
#
#         # GIN layers (protein)
#         nn1_xt = Sequential(Linear(num_features_xt, dim), ReLU(), Linear(dim, dim))
#         self.conv1_xt = GINConv(nn1_xt)
#         self.bn1_xt = torch.nn.BatchNorm1d(dim)
#
#         nn2_xt = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
#         self.conv2_xt = GINConv(nn2_xt)
#         self.bn2_xt = torch.nn.BatchNorm1d(dim)
#
#         nn3_xt = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
#         self.conv3_xt = GINConv(nn3_xt)
#         self.bn3_xt = torch.nn.BatchNorm1d(dim)
#
#         nn4_xt = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
#         self.conv4_xt = GINConv(nn4_xt)
#         self.bn4_xt = torch.nn.BatchNorm1d(dim)
#
#         nn5_xt = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
#         self.conv5_xt = GINConv(nn5_xt)
#         self.bn5_xt = torch.nn.BatchNorm1d(dim)
#
#         self.fc1_xt = Linear(dim, output_dim)
#
#         # dense
#         self.classifier = nn.Sequential(
#             nn.Linear(256, 1024),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(1024, 256),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(256, n_output),
#         )
#
#     def forward(self, data):
#         drug, target, y = data
#         xd = drug.x
#         xt, xt_ei, xt_batch = target.x, target.edge_index, target.batch
#
#         # drug
#         embedded_xd = self.embedding_xd(xd)
#         conv_xd = self.conv_xd_1(embedded_xd)
#         xd = self.fc1_xd(conv_xd.view(-1, 32 * 121))
#
#         # protein
#         xt = F.relu(self.conv1_xt(xt, xt_ei))
#         xt = self.bn1_xt(xt)
#         xt = F.relu(self.conv2_xt(xt, xt_ei))
#         xt = self.bn2_xt(xt)
#         xt = F.relu(self.conv3_xt(xt, xt_ei))
#         xt = self.bn3_xt(xt)
#         xt = F.relu(self.conv4_xt(xt, xt_ei))
#         xt = self.bn4_xt(xt)
#         xt = F.relu(self.conv5_xt(xt, xt_ei))
#         xt = self.bn5_xt(xt)
#         xt = gap(xt, xt_batch)
#         xt = F.relu(self.fc1_xt(xt))
#         xt = F.dropout(xt, p=0.2, training=self.training)
#
#         # joint
#         xj = torch.cat((xd, xt), 1)
#
#         # dense
#         out = self.classifier(xj).squeeze(1)
#         return out, y
#
#
# # d1 (graph) & d2 (graph) - DGraphDTA
# class GnG(torch.nn.Module):
#     def __init__(self, n_output=1, num_features_mol=55, num_features_pro=41,
#                  output_dim=128, dropout=0.2, joint='concat'):
#
#         super(GnG, self).__init__()
#
#         jdim = 256
#         self.joint = joint
#         if self.joint in ['add', 'multiple']:
#             jdim = 128
#
#         # drug
#         self.mol_conv1 = GCNConv(num_features_mol, num_features_mol)
#         self.mol_conv2 = GCNConv(num_features_mol, num_features_mol * 2)
#
#         # protein
#         self.pro_conv1 = GCNConv(num_features_pro, num_features_pro)
#         self.pro_conv2 = GCNConv(num_features_pro, num_features_pro * 2)
#
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(dropout)
#
#         # joint
#         if 'att' in self.joint:
#             self.mol_conv3 = GCNConv(num_features_mol * 2, output_dim)
#             self.pro_conv3 = GCNConv(num_features_pro * 2, output_dim)
#             self.jc = weight_norm(
#                 BANLayer(v_dim=output_dim, q_dim=output_dim, h_dim=jdim, h_out=2), name='h_mat', dim=None)
#         else:
#             self.mol_conv3 = GCNConv(num_features_mol * 2, num_features_mol * 4)
#             self.pro_conv3 = GCNConv(num_features_pro * 2, num_features_pro * 4)
#
#             # linear
#             self.mol_fc_g1 = torch.nn.Linear(num_features_mol * 4, 1024)
#             self.mol_fc_g2 = torch.nn.Linear(1024, output_dim)
#             self.pro_fc_g1 = torch.nn.Linear(num_features_pro * 4, 1024)
#             self.pro_fc_g2 = torch.nn.Linear(1024, output_dim)
#
#             if self.joint in ['concat', 'add', 'multiple']:
#                 self.jc = Simple_Joint(self.joint)
#             elif self.joint == 'bi':
#                 self.jc = Bilinear_Joint(output_dim, jdim)
#             else:
#                 raise Exception(f'{self.joint} method not supported!!!')
#
#         # dense
#         self.classifier = nn.Sequential(
#             nn.Linear(jdim, 1024),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(1024, 512),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(512, n_output),
#         )
#
#     def forward(self, data):
#         drug, target, y = data
#         xd, xd_ei, xd_batch = drug.x, drug.edge_index, drug.batch
#         xt, xt_ei, xt_batch = target.x, target.edge_index, target.batch
#
#         # drug
#         xd = self.relu(self.mol_conv1(xd, xd_ei))
#         xd = self.relu(self.mol_conv2(xd, xd_ei))
#         xd = self.relu(self.mol_conv3(xd, xd_ei))
#
#         # protein
#         xt = self.relu(self.pro_conv1(xt, xt_ei))
#         xt = self.relu(self.pro_conv2(xt, xt_ei))
#         xt = self.relu(self.pro_conv3(xt, xt_ei))
#
#         # joint
#         if 'att' in self.joint:
#             batch_size = len(y)
#             xd = xd.view(batch_size, -1, 128)
#             xt = xt.view(batch_size, -1, 128)
#             xj = self.jc(xd, xt)
#         else:
#             # flatten
#             xd = gep(xd, xd_batch)  # global pooling
#             xd = self.dropout(self.relu(self.mol_fc_g1(xd)))
#             xd = self.dropout(self.mol_fc_g2(xd))
#
#             xt = gep(xt, xt_batch)  # global pooling
#             xt = self.dropout(self.relu(self.pro_fc_g1(xt)))
#             xt = self.dropout(self.pro_fc_g2(xt))
#
#             xj = self.jc(xd, xt)
#
#         # dense
#         out = self.classifier(xj).squeeze(1)
#         return out, y


########################################################################################################################
########## Layers
########################################################################################################################


class Simple_Joint(nn.Module):
    def __init__(self, how):
        super(Simple_Joint, self).__init__()
        self.how = how

    def forward(self, xd, xt):
        if self.how == 'concat':
            return torch.cat((xd, xt), 1) # batch, 256
        elif self.how == 'add':
            return xd + xt
        elif self.how == 'multiple':
            return xd * xt
        else:
            raise Exception(f'how {self.how} not supported')


class Bilinear_Joint(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Bilinear_Joint, self).__init__()
        self.bilinear = nn.Bilinear(in_channels, in_channels, out_channels)
        self.relu = nn.ReLU()

    def forward(self, xd, xt):
        return self.relu(self.bilinear(xd, xt))


# AttentionMGT-DTA (Joint_Attention_Module)
class Joint_Attention_Module(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Joint_Attention_Module, self).__init__()

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.joint_attn_comp = nn.Linear(in_channels, in_channels)
        self.joint_attn_prot = nn.Linear(in_channels, in_channels)
        self.fc = nn.Linear(in_channels, out_channels)

    def forward(self, xd, xt):
        # compound-protein interaction
        inter_comp_prot = self.sigmoid(torch.einsum('bij,bkj->bik', self.joint_attn_prot(self.relu(xt)),
                                                    self.joint_attn_comp(self.relu(xd))))  # batch, xd, xt
        inter_comp_prot_sum = torch.einsum('bij->b', inter_comp_prot)  # batch, 1
        inter_comp_prot = torch.einsum('bij,b->bij', inter_comp_prot, 1 / inter_comp_prot_sum)  # batch, xd, xt

        # compound-protein joint embedding
        cp_embedding = self.tanh(torch.einsum('bij,bkj->bikj', xd, xt))  # batch, xd, xt, 128
        cp_embedding = torch.einsum('bijk,bij->bk', cp_embedding, inter_comp_prot)  # batch, 128
        cp_embedding = self.fc(cp_embedding)
        return cp_embedding


# for batch in val_loader:
#     break


# tmp = copy.deepcopy(batch)
# sns = SnS(joint='co_att')
# p, r = sns(tmp)
# print(p.shape, r.shape)
# sum(p.numel() for p in sns.parameters())

# tmp = copy.deepcopy(batch)
# sns = GnS(joint='bi_att')
# p, r = sns(tmp)
# print(p.shape, r.shape)
# sum(p.numel() for p in sns.parameters())

# tmp = copy.deepcopy(batch)
# sns = GnG(joint='att')
# p, r = sns(tmp)
# print(p.shape, r.shape)


