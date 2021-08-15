import numpy as np
import argparse
import torch
np.random.seed(0)
torch.autograd.set_detect_anomaly(True)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

''' Metapaths associated with the target datasets '''
meta_path_dict = {"acm":'pap,paiap,psp,pvp,pp', "slap":'gdcdg,gdg,gg,gog,gpg,gtg', "imdbc":'MAM,MDM', "imdbl":'0,1,2',
                  "dblp":'apa,apapa,apcpa,aptpa', "flickr":'uu,utu', "amazon":"copurchase,coview,similar"}

def parse_args():
    parser = argparse.ArgumentParser(description='SS-DCMultiplex')
    parser.add_argument('--embedder', nargs='?', default='SS-DCMultiplex', help='Which algorithm to use to embed')
    parser.add_argument('--dataset', nargs='?', default='imdbc', help='Dataset name')
    parser.add_argument('--metapaths', nargs='?', help='Relational aspects')

    parser.add_argument('--nb_epochs', type=int, default=10000, help='Number of iterations for training')
    parser.add_argument('--patience', type=int, default=35, help='Patience iterations for stopping training')
    parser.add_argument('--full_run', type=bool, default=True, help='Check test performance iteratively during training')
    parser.add_argument('--hid_units', type=int, default=64, help='Number of hidden units for node representation learning')
    parser.add_argument('--sc', type=float, default=3.0, help='GCN self connection weights')
    parser.add_argument('--n_layers', type=int, default=1, help='Number of hidden layers for node representation learning')
    parser.add_argument('--nheads', type=int, default=1, help='Number of attention heads for layer aggregation')
    parser.add_argument('--c_hid_units', type=int, default=64, help='Number of hidden units for cluster representation learning')
    parser.add_argument('--clusters', type=int, default=-1, help='Number of clusters; -1 if #Labels else otherwise')
    parser.add_argument('--activation', nargs='?', default='prelu', help='Activation function for node representation learning')
    parser.add_argument('--c_activation', nargs='?', default='softmax', help='Activation function for cluster membership learning')
    parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate')
    parser.add_argument('--drop_prob', type=float, default=0.5, help='Dropout probability')

    parser.add_argument('--incl_attr', type=bool, default=True, help='(0/1) Attributed or non-attributed network?')
    parser.add_argument('--cons', type=bool, default=True, help='(0/1) Learn consensus embeddings or not?')
    parser.add_argument('--isBias', type=bool, default=False, help='(0/1) Include bias component?')
    parser.add_argument('--isAttn', type=bool, default=True, help='(0/1) Include attention-based node/cluster representation aggregation?')
    parser.add_argument('--isSemi', type=bool, default=True, help='(0/1) Include supervision component?')

    parser.add_argument('--reg_coef', type=float, default=0.0001, help='Regularization co-efficient for discriminator')
    parser.add_argument('--sup_coef', type=float, default=0.5, help='Semi-Supervision co-efficient')
    parser.add_argument('--c_assign_coef', type=float, default=1.0, help='Cluster assignment co-efficient')
    parser.add_argument('--c_learn_coef', type=float, default=1.0, help='Cluster learning co-efficient')
    parser.add_argument('--n_cross_nw_coef', type=float, default=0.001, help='Node cross-network regularization co-efficient')
    parser.add_argument('--l2_coef', type=float, default=0.0001, help='L2 regularization co-efficient')

    parser.add_argument('--gpu_num', type=int, default=0, help='Which gpu (#id) to use? -1 if cpu')

    return parser.parse_known_args()

def printConfig(args):
    ''' Printing command-line arguments '''
    args_names = []
    args_vals = []
    for arg in vars(args):
        args_names.append(arg)
        args_vals.append(getattr(args, arg))
    print(args_names)
    print(args_vals)

def main():
    args, unknown = parse_args()
    args.metapaths = meta_path_dict[args.dataset]
    printConfig(args)

    from models.ssdcm import ssdcm
    model = ssdcm(args)

    model.training()

if __name__ == '__main__':
    main()
