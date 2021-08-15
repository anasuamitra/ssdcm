import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict

from layers.clustering_layer import CLayer
from layers.discriminator import Discriminator
from layers.attention import Attention
from layers.logreg import LogReg

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class modeler(nn.Module):
    def __init__(self, args):
        super(modeler, self).__init__()
        self.args = args

        if args.n_layers >= 2:
            from layers.gcn_res import GCN
        else:
            from layers.gcn import GCN
        self.gcn = nn.ModuleList(
            [GCN(args.ft_size, args.hid_units, args.n_layers, args.activation, args.drop_prob, args.isBias) for _ in range(args.nb_graphs)])

        self.clayer = nn.ModuleList([CLayer(args.hid_units, args.c_hid_units, args.nb_graphs, args.c_activation, args.drop_prob, args.isBias) for _ in range(args.nb_graphs)])

        self.disc = Discriminator(args.hid_units)

        self.H = [None] * args.nb_graphs
        self.Z = nn.ParameterList()
        for i in range(args.nb_graphs):
            self.Z.append(nn.Parameter(torch.FloatTensor(args.batch_size, args.clusters, args.hid_units)))
        self.U = nn.Parameter(torch.FloatTensor(args.batch_size, args.nb_nodes, args.hid_units))

        self.attn = nn.ModuleList([Attention(args)for _ in range(args.nheads)])
        self.logistic = LogReg(args.hid_units, args.nb_classes).to(args.device)
        self.act = nn.LogSigmoid()

        self.init_weight()

    def init_weight(self):
        nn.init.xavier_normal_(self.U)
        for i in range(self.args.nb_graphs):
            nn.init.xavier_normal_(self.Z[i])

    def forward(self, feature, adj, shuf, A, I, sparse, epoch, msk, samp_bias1, samp_bias2):
        u_1_all = []; u_2_all = []; h_1_all = []; c_1_all = []
        logits = []; result = {}; cross_reg_loss = 0.0
        h1_a_loss, h1_l_loss, h1_o_loss, c1_loss, u1_loss = 0.0, 0.0, 0.0, 0.0, 0.0

        for i in range(self.args.nb_graphs):
            a =  adj[i].to(self.args.device)
            f =  feature[i].to(self.args.device)
            sh = shuf[i].to(self.args.device)
            ''' Node representation learning. '''
            u_1 = self.gcn[i](f, a, sparse)
            u_2 = self.gcn[i](sh, a, sparse)
            ''' Cluster representation learning. '''
            self.H[i], self.Z[i] = self.clayer[i](u_1, self.Z[i], sparse)
            ''' Cluster-aware graph summary generation & Pooling. '''
            s = torch.unsqueeze(torch.spmm(torch.squeeze(self.H[i], 0), torch.squeeze(self.Z[i], 0)), 0)
            ''' Discriminator learning. '''
            logit = self.disc(s, u_1, u_2, samp_bias1, samp_bias2)

            u_1_all.append(u_1); u_2_all.append(u_2)
            logits.append(logit)

            D = torch.diagflat(torch.sum(A[0], dim=1))
            t_X = (D - A[0]).to(self.args.device)
            h1_l_loss += torch.trace(torch.spmm(torch.spmm(torch.squeeze(self.H[i], 0).t(), t_X), torch.squeeze(self.H[i], 0)))
            cluster_loss = (torch.squeeze(self.H[i], 0) * torch.squeeze(self.H[i], 0)).sum(1)  # inner product
            h1_o_loss += - self.act(cluster_loss).mean()
            # HTH = torch.unsqueeze(torch.spmm(torch.squeeze(self.H[i], 0).t(), torch.squeeze(self.H[i], 0)), 0)
            # h1_o_loss += ((HTH - I) ** 2).sum()

        result['logits'] = logits
        result['rel_clus_learning_loss'] = h1_l_loss
        result['rel_clus_assignment_loss'] = h1_o_loss

        ''' Cross-graph node and cluster regularization '''
        if self.args.n_cross_nw_coef != 0:
            for i in range(self.args.nb_graphs):
                for j in range(self.args.nb_graphs):
                    if (i != j):
                        cross_reg_loss += self.args.n_cross_nw_coef * ((torch.squeeze(u_1_all[i], 0) - 1.0 * torch.squeeze(u_1_all[j], 0)) ** 2).sum()
                        cross_reg_loss += self.args.n_cross_nw_coef * ((torch.squeeze(self.H[i], 0) - 1.0 * torch.squeeze(self.H[j], 0)) ** 2).sum()
            result['cross_reg_loss'] = cross_reg_loss

        if self.args.isAttn:
            u_1_all_lst = []; u_2_all_lst = []
            for h_idx in range(self.args.nheads):
                u_1_all_, u_2_all_, n_attn = self.attn[h_idx](u_1_all, u_2_all)
                u_1_all_lst.append(u_1_all_); u_2_all_lst.append(u_2_all_)
            u_1_comb = torch.mean(torch.cat(u_1_all_lst, 0), 0).unsqueeze(0)
            u_2_comb = torch.mean(torch.cat(u_2_all_lst, 0), 0).unsqueeze(0)
        else:
            u_1_comb = torch.mean(torch.cat(u_1_all), 0).unsqueeze(0)
            u_2_comb = torch.mean(torch.cat(u_2_all), 0).unsqueeze(0)

        ''' Consensus regularization '''
        if self.args.cons:
            pos_reg_loss = ((self.U - u_1_comb) ** 2).sum()
            neg_reg_loss = ((self.U - u_2_comb) ** 2).sum()
            reg_loss = pos_reg_loss - neg_reg_loss
            result['reg_loss'] = reg_loss
        else:
            pos_reg_loss = ((self.U - u_1_comb) ** 2).sum()
            result['reg_loss'] = pos_reg_loss

        ''' Incorporating supervision information '''
        if self.args.isSemi:
            semi = self.logistic(self.U).squeeze(0)
            result['semi'] = semi

        ''' Logging related information '''
        loss_dict = OrderedDict()
        loss_dict['rel_clus_assignment_loss'] = result['rel_clus_assignment_loss']
        loss_dict['rel_clus_learning_loss'] = result['rel_clus_learning_loss']
        loss_dict['logits'] = sum(result['logits'])/self.args.nb_graphs
        loss_dict['reg_loss'] = result['reg_loss']
        if self.args.n_cross_nw_coef != 0:
            loss_dict['cross_reg_loss'] = result['cross_reg_loss']
        if self.args.isSemi:
            loss_dict['semi'] = result['semi']

        return result

