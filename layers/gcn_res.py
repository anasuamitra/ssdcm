import torch
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import torch.nn as nn
import torch.nn.functional as F
import pdb
import math

''' Relation specific Graph Convolutional Neural Network with dropouts, skip-connections and prelu activation. '''

class GCNLayer(nn.Module):
    def __init__(self, in_ft, out_ft, act='prelu', drop_prob=0.5, isBias=False, residual=True):
        super(GCNLayer, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        if isBias:
            self.bias_1 = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias_1.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)
        self.drop_prob = drop_prob
        self.isBias = isBias
        if act == 'prelu':
            self.act = nn.PReLU()
        else:
            self.act = None
        self.residual = residual
        if self.residual:
            self.res_fc = nn.Linear(in_ft, out_ft, bias=False)
        else:
            self.res_fc = None
        self.weights_init()
        # self.reset_parameters()

    def weights_init(self):
        torch.nn.init.xavier_uniform_(self.fc.weight.data, gain=1.414)
        if self.fc.bias is not None:
            self.fc.bias.data.fill_(0.0)
        nn.init.xavier_normal_(self.res_fc.weight.data, gain=1.414)

    def reset_parameters(self):
        # pdb.set_trace()
        stdv = 1. / math.sqrt(self.fc_1.weight.data.size(0))
        self.fc_1.weight.data.uniform_(-stdv, stdv)
        if self.bias_1 is not None:
            self.bias_1.data.uniform_(-stdv, stdv)

    def forward(self, seq, adj, sparse=False):
        feat = seq
        seq = F.dropout(seq, self.drop_prob, training=self.training)
        seq = self.fc(seq)
        if sparse:
            seq = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq, 0)), 0)
        else:
            seq = torch.bmm(adj, seq)
        if self.isBias:
            seq += self.bias_1

        if self.residual:
            if self.res_fc is not None:
                resval = self.res_fc(feat)
                seq = 0.2 * resval + 0.8 * seq

        if self.act is not None:
            seq = self.act(seq)

        return seq

class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, n_layers = 3, act='prelu', drop_prob=0.5, isBias=False):
        super(GCN, self).__init__()
        assert n_layers >= 2
        self.n_layers = n_layers
        self.layers = nn.ModuleList()
        # input layer self, in_ft, out_ft, act='prelu', drop_prob=0.5, isBias=False
        self.layers.append(GCNLayer(in_ft, out_ft, act, drop_prob, isBias))
        # hidden layers
        for l in range(n_layers - 2):
            self.layers.append(GCNLayer(out_ft, out_ft, act, drop_prob, isBias))
        # output layer
        self.layers.append(GCNLayer(out_ft, out_ft, act='prelu', drop_prob=0.5, isBias=False))

    def forward(self, seq, adj, sparse=False):
        for layer in self.layers:
            seq = layer(seq, adj, sparse)

        return seq
