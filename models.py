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
            elif self.joint == 'joint_att':
                self.jc = Joint_Attention_Module(107, jdim)
            elif self.joint == 'multi_att':
                self.jc = Multi_Head_Attention()
            else:
                raise Exception('wrong att type')

        else:
            self.jc = Simple_Joint(self.joint)
            if self.joint != 'concat':
                jdim = 128

        # 1D convolution on smiles sequence
        self.embedding_xd = nn.Embedding(num_features_xd + 1, embed_dim) # batch, 100, 128
        self.conv_xd_1 = nn.Conv1d(in_channels=100, out_channels=n_filters, kernel_size=kd_size) # batch, 32, 125
        self.conv_xd_2 = nn.Conv1d(in_channels=n_filters, out_channels=n_filters * 2, kernel_size=kd_size) # batch, 64, 122
        self.conv_xd_3 = nn.Conv1d(in_channels=n_filters * 2, out_channels=n_filters * 3, kernel_size=kd_size) # batch, 96, 119

        # 1D convolution on protein sequence
        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim) # batch, 1000, 128
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


# d1 (graph) & d2 (seq) - GraphDTA (att-DrugBAN)
class GnS(torch.nn.Module):
    def __init__(self, n_output=1, num_features_xd=55, num_features_xt=25,
                 n_filters=32, embed_dim=128, output_dim=128, dropout=0.2, joint='concat'):

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

        if 'att' in self.joint:
            nn5_xd = Sequential(Linear(dim, dim), ReLU(), Linear(dim, output_dim))
            self.conv5_xd = GINConv(nn5_xd)
            self.bn5_xd = torch.nn.BatchNorm1d(output_dim)

            self.protein_extractor = ProteinCNN(embedding_dim=128) # batch, 985, 128
            self.jc = weight_norm(BANLayer(v_dim=128, q_dim=128, h_dim=jdim, h_out=2), name='h_mat', dim=None)
        else:
            nn5_xd = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
            self.conv5_xd = GINConv(nn5_xd)
            self.bn5_xd = torch.nn.BatchNorm1d(dim)

            self.fc1_xd = Linear(dim, output_dim)

            # 1D convolution on protein sequence
            self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
            self.conv_xt_1 = nn.Conv1d(in_channels=1000, out_channels=n_filters, kernel_size=8)  # batch, 32, 121
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

        # joint
        if 'att' in self.joint:
            xd = xd.view(len(y), -1, 128)
            xt = self.protein_extractor(xt)
            xj = self.jc(xd, xt)
        else:
            # flatten
            xd = gap(xd, xd_batch)
            xd = F.relu(self.fc1_xd(xd))
            xd = F.dropout(xd, p=0.2, training=self.training)
            embedded_xt = self.embedding_xt(xt)
            conv_xt = self.conv_xt_1(embedded_xt)
            xt = self.fc1_xt(conv_xt.view(-1, 32 * 121))

            xj = self.jc(xd, xt)

        # dense
        out = self.classifier(xj).squeeze(1)
        return out, y


# d1 (seq) & d2 (graph) - Custom
class SnG(torch.nn.Module):
    def __init__(self, n_output=1, num_features_xd=64, num_features_xt=41,
                 n_filters=32, embed_dim=128, output_dim=128, dropout=0.2, joint='concat'):

        super(SnG, self).__init__()

        dim = 32
        self.joint = joint

        # 1D convolution on smiles sequence
        self.embedding_xd = nn.Embedding(num_features_xd + 1, embed_dim)
        self.conv_xd_1 = nn.Conv1d(in_channels=100, out_channels=n_filters, kernel_size=8)
        self.fc1_xd = nn.Linear(32 * 121, output_dim)

        # GIN layers (protein)
        nn1_xt = Sequential(Linear(num_features_xt, dim), ReLU(), Linear(dim, dim))
        self.conv1_xt = GINConv(nn1_xt)
        self.bn1_xt = torch.nn.BatchNorm1d(dim)

        nn2_xt = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv2_xt = GINConv(nn2_xt)
        self.bn2_xt = torch.nn.BatchNorm1d(dim)

        nn3_xt = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv3_xt = GINConv(nn3_xt)
        self.bn3_xt = torch.nn.BatchNorm1d(dim)

        nn4_xt = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv4_xt = GINConv(nn4_xt)
        self.bn4_xt = torch.nn.BatchNorm1d(dim)

        nn5_xt = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv5_xt = GINConv(nn5_xt)
        self.bn5_xt = torch.nn.BatchNorm1d(dim)

        self.fc1_xt = Linear(dim, output_dim)

        # dense
        self.classifier = nn.Sequential(
            nn.Linear(256, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_output),
        )

    def forward(self, data):
        drug, target, y = data
        xd = drug.x
        xt, xt_ei, xt_batch = target.x, target.edge_index, target.batch

        # drug
        embedded_xd = self.embedding_xd(xd)
        conv_xd = self.conv_xd_1(embedded_xd)
        xd = self.fc1_xd(conv_xd.view(-1, 32 * 121))

        # protein
        xt = F.relu(self.conv1_xt(xt, xt_ei))
        xt = self.bn1_xt(xt)
        xt = F.relu(self.conv2_xt(xt, xt_ei))
        xt = self.bn2_xt(xt)
        xt = F.relu(self.conv3_xt(xt, xt_ei))
        xt = self.bn3_xt(xt)
        xt = F.relu(self.conv4_xt(xt, xt_ei))
        xt = self.bn4_xt(xt)
        xt = F.relu(self.conv5_xt(xt, xt_ei))
        xt = self.bn5_xt(xt)
        xt = gap(xt, xt_batch)
        xt = F.relu(self.fc1_xt(xt))
        xt = F.dropout(xt, p=0.2, training=self.training)

        # joint
        xj = torch.cat((xd, xt), 1)

        # dense
        out = self.classifier(xj).squeeze(1)
        return out, y


# d1 (graph) & d2 (graph) - DGraphDTA
class GnG(torch.nn.Module):
    def __init__(self, n_output=1, num_features_mol=55, num_features_pro=41,
                 output_dim=128, dropout=0.2, joint='concat'):

        super(GnG, self).__init__()

        jdim = 256
        self.joint = joint
        if self.joint in ['add', 'multiple']:
            jdim = 128

        # drug
        self.mol_conv1 = GCNConv(num_features_mol, num_features_mol)
        self.mol_conv2 = GCNConv(num_features_mol, num_features_mol * 2)

        # protein
        self.pro_conv1 = GCNConv(num_features_pro, num_features_pro)
        self.pro_conv2 = GCNConv(num_features_pro, num_features_pro * 2)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # joint
        if 'att' in self.joint:
            self.mol_conv3 = GCNConv(num_features_mol * 2, output_dim)
            self.pro_conv3 = GCNConv(num_features_pro * 2, output_dim)
            self.jc = weight_norm(
                BANLayer(v_dim=output_dim, q_dim=output_dim, h_dim=jdim, h_out=2), name='h_mat', dim=None)
        else:
            self.mol_conv3 = GCNConv(num_features_mol * 2, num_features_mol * 4)
            self.pro_conv3 = GCNConv(num_features_pro * 2, num_features_pro * 4)

            # linear
            self.mol_fc_g1 = torch.nn.Linear(num_features_mol * 4, 1024)
            self.mol_fc_g2 = torch.nn.Linear(1024, output_dim)
            self.pro_fc_g1 = torch.nn.Linear(num_features_pro * 4, 1024)
            self.pro_fc_g2 = torch.nn.Linear(1024, output_dim)

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
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, n_output),
        )

    def forward(self, data):
        drug, target, y = data
        xd, xd_ei, xd_batch = drug.x, drug.edge_index, drug.batch
        xt, xt_ei, xt_batch = target.x, target.edge_index, target.batch

        # drug
        xd = self.relu(self.mol_conv1(xd, xd_ei))
        xd = self.relu(self.mol_conv2(xd, xd_ei))
        xd = self.relu(self.mol_conv3(xd, xd_ei))

        # protein
        xt = self.relu(self.pro_conv1(xt, xt_ei))
        xt = self.relu(self.pro_conv2(xt, xt_ei))
        xt = self.relu(self.pro_conv3(xt, xt_ei))

        # joint
        if 'att' in self.joint:
            batch_size = len(y)
            xd = xd.view(batch_size, -1, 128)
            xt = xt.view(batch_size, -1, 128)
            xj = self.jc(xd, xt)
        else:
            # flatten
            xd = gep(xd, xd_batch)  # global pooling
            xd = self.dropout(self.relu(self.mol_fc_g1(xd)))
            xd = self.dropout(self.mol_fc_g2(xd))

            xt = gep(xt, xt_batch)  # global pooling
            xt = self.dropout(self.relu(self.pro_fc_g1(xt)))
            xt = self.dropout(self.pro_fc_g2(xt))

            xj = self.jc(xd, xt)

        # dense
        out = self.classifier(xj).squeeze(1)
        return out, y


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


# AttentionDTA
class Multi_Head_Attention(nn.Module):
    def __init__(self, head = 8, conv=32, out_channels=128, device='cpu'):
        super(Multi_Head_Attention, self).__init__()
        self.conv = conv
        self.head = head
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.d_a = nn.Linear(self.conv * 3, self.conv * 3 * head)
        self.p_a = nn.Linear(self.conv * 3, self.conv * 3 * head)
        self.scale = torch.sqrt(torch.FloatTensor([self.conv * 3])).to(device)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(self.conv * 3, out_channels)

    def forward(self, drug, protein):
        bsz, d_ef,d_il = drug.shape
        bsz, p_ef, p_il = protein.shape
        drug_att = self.relu(self.d_a(drug.permute(0, 2, 1))).view(bsz, self.head, d_il, d_ef)
        protein_att = self.relu(self.p_a(protein.permute(0, 2, 1))).view(bsz, self.head, p_il, p_ef)
        interaction_map = torch.mean(self.tanh(torch.matmul(drug_att, protein_att.permute(0, 1, 3, 2)) / self.scale),1)
        Compound_atte = self.tanh(torch.sum(interaction_map, 2)).unsqueeze(1)
        Protein_atte = self.tanh(torch.sum(interaction_map, 1)).unsqueeze(1)
        drug = drug * Compound_atte
        protein = protein * Protein_atte

        drug = self.fc(self.pool(drug).view(-1, self.conv * 3))
        protein = self.fc(self.pool(protein).view(-1, self.conv * 3))
        emb = torch.cat((drug, protein), dim=1)
        return emb


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


# DrugBAN (extract protein repr)
class ProteinCNN(nn.Module):
    def __init__(self, embedding_dim, padding=True):
        super(ProteinCNN, self).__init__()
        if padding:
            self.embedding = nn.Embedding(26, embedding_dim, padding_idx=0)
        else:
            self.embedding = nn.Embedding(26, embedding_dim)
        in_ch = [128] + [128, 128, 128]

        self.in_ch = in_ch[-1]
        kernels = [3, 6, 9]
        self.conv1 = nn.Conv1d(in_channels=in_ch[0], out_channels=in_ch[1], kernel_size=kernels[0])
        self.bn1 = nn.BatchNorm1d(in_ch[1])
        self.conv2 = nn.Conv1d(in_channels=in_ch[1], out_channels=in_ch[2], kernel_size=kernels[1])
        self.bn2 = nn.BatchNorm1d(in_ch[2])
        self.conv3 = nn.Conv1d(in_channels=in_ch[2], out_channels=in_ch[3], kernel_size=kernels[2])
        self.bn3 = nn.BatchNorm1d(in_ch[3])

    def forward(self, v):
        v = self.embedding(v.long())
        v = v.transpose(2, 1)
        v = self.bn1(F.relu(self.conv1(v)))
        v = self.bn2(F.relu(self.conv2(v)))
        v = self.bn3(F.relu(self.conv3(v)))
        v = v.view(v.size(0), v.size(2), -1)
        return v


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
        if 1 < self.k:
            fusion_logits = fusion_logits.unsqueeze(1)  # b x 1 x d
            fusion_logits = self.p_net(fusion_logits).squeeze(1) * self.k  # sum-pooling
        return fusion_logits

    def forward(self, v, q, softmax=False):
        # print(v.shape, q.shape)
        # print('====================================')
        v_num = v.size(1) # v.size [batch_size(8), drug_representation = (290, hidden_dimension(128))]
        q_num = q.size(1) # q.size [batch_size(8), protein_representation = (1185, hidden_dimension(128))]
        if self.h_out <= self.c:
            v_ = self.v_net(v) # 8, 290, 768
            q_ = self.q_net(q) # 8, 1185, 768
            att_maps = torch.einsum('xhyk,bvk,bqk->bhvq', (self.h_mat, v_, q_)) + self.h_bias
            # print(self.h_mat.size()) # 1 2 1 768
            # print(self.h_bias.size()) # 1 2 1 1
            # print(att_maps.shape) # 8 2 290 1185

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
        # print(logits.shape) # 8, 256
        for i in range(1, self.h_out):
            logits_i = self.attention_pooling(v_, q_, att_maps[:, i, :, :])
            logits += logits_i
        logits = self.bn(logits)
        # return logits, att_maps
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

# for batch in val_loader:
#     break
#
# tmp = copy.deepcopy(batch)
# sns = SnS(joint='bi_att')
# p, r = sns(tmp)
# print(p.shape, r.shape)
# sum(p.numel() for p in sns.parameters())

# tmp = copy.deepcopy(batch)
# sns = GnS(joint='att')
# p, r = sns(tmp)
# gap(p, r).shape
# print(p.shape, r.shape)

# tmp = copy.deepcopy(batch)
# sns = GnG(joint='att')
# p, r = sns(tmp)
# print(p.shape, r.shape)
