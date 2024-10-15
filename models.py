########################################################################################################################
########## Sources
########################################################################################################################

# https://github.com/thinng/GraphDTA

########################################################################################################################
########## Import
########################################################################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, global_add_pool

########################################################################################################################
########## Models
########################################################################################################################


# d1 (seq) & d2 (seq)
class SnS(torch.nn.Module):
    def __init__(self, n_output=1, num_features_xd=64, num_features_xt=25,
                 n_filters=32, embed_dim=128, output_dim=128, dropout=0.2):

        super(SnS, self).__init__()

        # 1D convolution on smiles sequence
        self.embedding_xd = nn.Embedding(num_features_xd + 1, embed_dim)
        self.conv_xd_1 = nn.Conv1d(in_channels=150, out_channels=n_filters, kernel_size=8)
        self.fc1_xd = nn.Linear(32 * 121, output_dim)

        # 1D convolution on protein sequence
        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
        self.conv_xt_1 = nn.Conv1d(in_channels=1500, out_channels=n_filters, kernel_size=8)
        self.fc1_xt = nn.Linear(32 * 121, output_dim)

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
        xd, xt = drug.x, target.x

        # drug
        embedded_xd = self.embedding_xd(xd)
        conv_xd = self.conv_xd_1(embedded_xd)
        xd = self.fc1_xd(conv_xd.view(-1, 32 * 121))

        # protein
        embedded_xt = self.embedding_xt(xt)
        conv_xt = self.conv_xt_1(embedded_xt)
        xt = self.fc1_xt(conv_xt.view(-1, 32 * 121))

        # joint
        xj = torch.cat((xd, xt), 1)

        # dense
        out = self.classifier(xj).squeeze(1)
        return out, y


# d1 (graph) & d2 (seq)
class GnS(torch.nn.Module):
    def __init__(self, n_output=1, num_features_xd=55, num_features_xt=25,
                 n_filters=32, embed_dim=128, output_dim=128, dropout=0.2):

        super(GnS, self).__init__()

        dim = 32

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

        nn5_xd = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv5_xd = GINConv(nn5_xd)
        self.bn5_xd = torch.nn.BatchNorm1d(dim)

        self.fc1_xd = Linear(dim, output_dim)

        # 1D convolution on protein sequence
        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
        self.conv_xt_1 = nn.Conv1d(in_channels=1500, out_channels=n_filters, kernel_size=8)
        self.fc1_xt = nn.Linear(32*121, output_dim)

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
        xd = global_add_pool(xd, xd_batch)
        xd = F.relu(self.fc1_xd(xd))
        xd = F.dropout(xd, p=0.2, training=self.training)

        # protein
        embedded_xt = self.embedding_xt(xt)
        conv_xt = self.conv_xt_1(embedded_xt)
        xt = conv_xt.view(-1, 32 * 121)
        xt = self.fc1_xt(xt)

        # joint
        xj = torch.cat((xd, xt), 1)

        # dense
        out = self.classifier(xj).squeeze(1)
        return out, y


# d1 (seq) & d2 (graph)
class SnG(torch.nn.Module):
    def __init__(self, n_output=1, num_features_xd=64, num_features_xt=41,
                 n_filters=32, embed_dim=128, output_dim=128, dropout=0.2):

        super(SnG, self).__init__()

        dim = 32

        # 1D convolution on smiles sequence
        self.embedding_xd = nn.Embedding(num_features_xd + 1, embed_dim)
        self.conv_xd_1 = nn.Conv1d(in_channels=150, out_channels=n_filters, kernel_size=8)
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
        xt = global_add_pool(xt, xt_batch)
        xt = F.relu(self.fc1_xt(xt))
        xt = F.dropout(xt, p=0.2, training=self.training)

        # joint
        xj = torch.cat((xd, xt), 1)

        # dense
        out = self.classifier(xj).squeeze(1)
        return out, y


# d1 (graph) & d2 (graph)
class GnG(torch.nn.Module):
    def __init__(self, n_output=1,num_features_xd=55, num_features_xt=41, output_dim=128, dropout=0.2):

        super(GnG, self).__init__()

        dim = 32

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

        nn5_xd = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv5_xd = GINConv(nn5_xd)
        self.bn5_xd = torch.nn.BatchNorm1d(dim)

        self.fc1_xd = Linear(dim, output_dim)

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
        xd, xd_ei, xd_batch = drug.x, drug.edge_index, drug.batch
        xt, xt_ei, xt_batch = target.x, target.edge_index, target.batch

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
        xd = global_add_pool(xd, xd_batch)
        xd = F.relu(self.fc1_xd(xd))
        xd = F.dropout(xd, p=0.2, training=self.training)

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
        xt = global_add_pool(xt, xt_batch)
        xt = F.relu(self.fc1_xt(xt))
        xt = F.dropout(xt, p=0.2, training=self.training)

        # joint
        xj = torch.cat((xd, xt), 1)

        # dense
        out = self.classifier(xj).squeeze(1)
        return out, y