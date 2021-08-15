import torch
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import torch.nn as nn
import torch.nn.functional as F
import pdb
import math

''' Cluster-pooling over GCN, inner-product based projection of node and cluster representations
to learn softmax-based cluster membership. Use of dropout improves overall performance. '''

class CLayer(nn.Module):
    def __init__(self, out_ft, c_out_ft, nb_clus, act='softmax', drop_prob=0.5, isBias=False):
        super(CLayer, self).__init__()

        if act == 'softmax':
            self.act = nn.Softmax(dim=-1)
        else:
            self.act = None

        # self.fc_3 = nn.Linear(c_out_ft, nb_clus)

        if isBias:
            self.bias_1 = nn.Parameter(torch.FloatTensor(c_out_ft))
            self.bias_1.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

        self.isBias = isBias
        self.drop_prob = drop_prob
        # self.reset_parameters()
        # print(next(self.parameters()).device)

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

    def forward(self, seq, c_rprs, sparse=False):
        if sparse:
            seq = torch.unsqueeze(torch.spmm(torch.squeeze(seq, 0), torch.squeeze(c_rprs, 0).t()), 0) # [1, 3550, 20] = [1, 3550, 64] x [1, 64, 20]
        else:
            seq = torch.bmm(seq, c_rprs.t()) # [1, 3550, 20] = [1, 3550, 64] x [1, 64, 3]

        if self.isBias:
            seq += self.bias_1

        return torch.unsqueeze(self.act(torch.squeeze(seq, 0)), 0), c_rprs # [1, 3550, 3], [1, 3, 64]