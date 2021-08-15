import torch
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import torch.nn as nn
import torch.nn.functional as F
import pdb
import math

''' Relation specific Graph Convolutional Neural Network with dropouts and prelu activation. '''

class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, n_layers=1, act='prelu', drop_prob=0.5, isBias=False):
        super(GCN, self).__init__()
        self.fc_1 = nn.Linear(in_ft, out_ft, bias=False)

        if act == 'prelu':
            self.act = nn.PReLU()
        else:
            self.act = None

        if isBias:
            self.bias_1 = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias_1.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

        self.drop_prob = drop_prob
        self.isBias = isBias
        # self.reset_parameters()

    def reset_parameters(self):
        # pdb.set_trace()
        stdv = 1. / math.sqrt(self.fc_1.weight.data.size(0))
        self.fc_1.weight.data.uniform_(-stdv, stdv)
        if self.bias_1 is not None:
            self.bias_1.data.uniform_(-stdv, stdv)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq, adj, sparse=False):
        seq = F.dropout(seq, self.drop_prob, training=self.training) # Shape of seq: (batch, nodes, features) = (([1, 3550, 2000]))
        seq = self.fc_1(seq) # Shape of seq: (batch, nodes, features) = (([1, 3550, 64]))

        if sparse:
            seq = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq, 0)), 0) # (1 x [3550, 3550] x [3550, 64] = [1, 3550, 64])
        else:
            seq = torch.bmm(adj, seq)

        if self.isBias:
            seq += self.bias_1

        if self.act is not None:
            seq = self.act(seq)

        return seq # ([1, 3550, 64])

