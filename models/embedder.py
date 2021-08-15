import numpy as np
import torch

from utils import data_utils

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class embedder():
    ''' All important experiment settings. Pre-processing of dataset. '''
    def __init__(self, args):
        args.batch_size = 1
        args.sparse = True
        args.metapaths = args.metapaths.split(",")
        if args.gpu_num == -1:
            args.device = 'cpu'
        else:
            args.device = torch.device("cuda:" + str(args.gpu_num) if torch.cuda.is_available() else "cpu")

        adj, adj_list, features, labels, idx_train, idx_val, idx_test = data_utils.load_data(args)
        features = [data_utils.preprocess_features(feature) for feature in features]
        adj = [data_utils.normalize_adj(adj_) for adj_ in adj]
        for i in range(len(adj)):
            adj_list[i] = adj_list[i].to(args.device)

        args.nb_nodes = features[0].shape[0]
        args.ft_size = features[0].shape[1]
        args.nb_classes = labels.shape[1]
        args.nb_graphs = len(adj)
        if args.clusters == -1:
            args.clusters = args.labels
        ''' Get cluster kernel and identity(k) matrix. '''
        self.A = data_utils.get_cluster_kernel(args.nb_nodes, args.nb_classes, idx_train, labels)
        self.I = torch.eye(args.clusters).reshape((1, args.clusters, args.clusters)).to(args.device)

        self.adj = [data_utils.sparse_mx_to_torch_sparse_tensor(adj_) for adj_ in adj] # Shape: (batch, nodes, nodes) = (([1, 3550, 3550]))
        self.features = [torch.FloatTensor(feature[np.newaxis]) for feature in features] # Shape: (batch, nodes, features) = (([1, 3550, 2000]))
        self.labels = torch.FloatTensor(labels[np.newaxis]).to(args.device) # Shape: (batch, nodes, labels) = (([1, 3550, 3]))
        self.idx_train = torch.LongTensor(idx_train).to(args.device)
        self.idx_val = torch.LongTensor(idx_val).to(args.device)
        self.idx_test = torch.LongTensor(idx_test).to(args.device)
        self.train_lbls = torch.argmax(self.labels[0, self.idx_train], dim=1)
        self.val_lbls = torch.argmax(self.labels[0, self.idx_val], dim=1)
        self.test_lbls = torch.argmax(self.labels[0, self.idx_test], dim=1)

        self.args = args