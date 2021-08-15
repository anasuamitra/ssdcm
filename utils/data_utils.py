import numpy as np
import pickle as pkl
import scipy.sparse as sp
import torch

''' Fetch and preprocess dataset '''
def load_data(args):
    dataset = args.dataset
    metapaths = args.metapaths
    sc = args.sc
    data = pkl.load(open('data/{}.pkl'.format(dataset), "rb"))

    label = data['labels']
    N = label.shape[0]
    if args.incl_attr == 1:
        truefeatures = data['features'].astype(float)
        truefeatures = sp.lil_matrix(truefeatures)
    rownetworks = [data["layers"][metapath] + np.eye(N) * sc for metapath in metapaths]
    rownetworks = [sp.csr_matrix(rownetwork) for rownetwork in rownetworks]
    idx_train = data["splits"]["-1"]['train_idx'].ravel()
    idx_val = data["splits"]["-1"]['val_idx'].ravel()
    idx_test = data["splits"]["-1"]['test_idx'].ravel()

    truefeatures_list = []
    if args.incl_attr:
        for _ in range(len(rownetworks)):
            truefeatures_list.append(truefeatures)
    elif not args.incl_attr:
        truefeatures_list = rownetworks

    adj_list = list()
    for i in range(len(rownetworks)):
        row, col, data = list(), list(), list()
        row.extend(rownetworks[i].tocoo().row)
        col.extend(rownetworks[i].tocoo().col)
        data.extend(rownetworks[i].tocoo().data)
        assert len(row) == len(col) == len(data)
        c = torch.LongTensor(np.vstack((np.array(row), np.array(col))))
        adj_list.append(c)

    return rownetworks, adj_list, truefeatures_list, label, idx_train, idx_val, idx_test


''' Get Semi-Supervised Cluster Similarity Kernel. '''
def get_cluster_kernel(nb_nodes, nb_classes, idx_train, labels):
    A = list() # List of SS-Cluster Kernels for each relation
    WY = sp.lil_matrix(np.zeros((nb_nodes, nb_classes))) # W is a penalty matrix
    WY[idx_train, :] = labels[idx_train, : ] # Filtering out test-label information
    WYW = WY.dot(WY.transpose()) # Label similarity kernel based on train-points
    A.append(torch.FloatTensor(WYW.todense()))

    return A

###############################################
# This section of code adapted from tkipf/gcn #
###############################################
def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    with np.errstate(divide='ignore'):
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_inv[np.isnan(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense()


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    with np.errstate(divide='ignore'):
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_inv_sqrt[np.isnan(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    # a = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    a = adj.dot(d_mat_inv_sqrt)
    a = d_mat_inv_sqrt.dot(a)
    return a.tocoo()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

