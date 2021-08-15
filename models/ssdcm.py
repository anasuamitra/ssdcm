import numpy as np
import pandas as pd
from collections import OrderedDict

from models.embedder  import embedder
from models.modeler import modeler

import torch
import torch.nn as nn
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class ssdcm(embedder):
    def __init__(self, args):
        ''' Initializing class and fetching dataset, model-specific arguments. '''
        embedder.__init__(self, args)
        self.args = args

    def writetofile_fn(self, results):
        ''' Script to dump algorithm performance in results/ folder '''
        columns = ["Algorithm", "Dataset", "incl_attr", "cons", "reg_coef", "sup_coef", "c_assign_coef",
                   "n_cross_nw_coef", "l2_coef", "lr", "n_layers", "nheads", "pool_type", "isSemi", "isAttn", "clusters"]
        result_keys = ["micro_f1", "macro_f1", "nmi", "nmi_h", "S@5", "S@10", "S@20", "S@50", "S@100"]
        for i in result_keys:
            columns += [i]
        for r in range(self.args.nb_graphs):
            name = "nmi_h" + str(r)
            columns += [name]
        results_df = pd.DataFrame(columns=columns)
        temp = OrderedDict()
        temp["Algorithm"] = self.args.embedder
        temp["Dataset"] = self.args.dataset
        temp["incl_attr"] = self.args.incl_attr
        temp["cons"] = self.args.cons
        temp["reg_coef"] = self.args.reg_coef
        temp["sup_coef"] = self.args.sup_coef
        temp["c_assign_coef"] = self.args.c_assign_coef
        temp["n_cross_nw_coef"] = self.args.n_cross_nw_coef
        temp["l2_coef"] = self.args.l2_coef
        temp["lr"] = self.args.lr
        temp["n_layers"] = self.args.n_layers
        temp["nheads"] = self.args.nheads
        temp["pool_type"] = self.args.pool_type
        temp["isSemi"] = self.args.isSemi
        temp["isAttn"] = self.args.isAttn
        temp["clusters"] = self.args.clusters
        for i in result_keys:
            temp[i] = results[i]
        for r in range(self.args.nb_graphs):
            name = "nmi_h" + str(r)
            temp[name] = results[name]
        results_df = results_df.append(temp, ignore_index=True)
        with open("results/" + self.args.embedder + "_" + self.args.dataset + ".csv", 'a') as file:
            results_df.to_csv(file, index=False, header=file.tell() == 0)

    def training(self):
        ''' Build all the components of the architecture and initialize its parameters. '''
        model = modeler(self.args)
        model = model.to(self.args.device)

        ''' Printing all the valid parameters of the constructed architecture. '''
        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         print(name, param.data, param.data.shape, param.device)

        optimiser = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=self.args.l2_coef) # Initialize training method
        cnt_wait = 0; best = 1e14 # Train stopping criteria
        b_xent = nn.BCEWithLogitsLoss().to(self.args.device) # Discriminator Loss
        xent = nn.CrossEntropyLoss().to(self.args.device) # SSL Loss
        best_results = dict(); best_count = 0; best_sup = None

        for epoch in range(self.args.nb_epochs):
            xent_loss = None
            model.train() # Tell PyTorch that we are training through model object.
            optimiser.zero_grad() # Reset gradients.

            ''' SHuffling attributes to generate negative local patches. '''
            idx = np.random.permutation(self.args.nb_nodes)
            shuf = [feature[:, idx, :] for feature in self.features]

            lbl_1 = torch.ones(self.args.batch_size, self.args.nb_nodes) # Positive labels for samples
            lbl_2 = torch.zeros(self.args.batch_size, self.args.nb_nodes) # Negative labels for samples
            lbl = torch.cat((lbl_1, lbl_2), 1).to(self.args.device)

            result = model(self.features, self.adj, shuf, self.A, self.I, self.args.sparse, epoch, None, None, None)

            logits = result['logits']
            for view_idx, logit in enumerate(logits):
                if xent_loss is None:
                    xent_loss = b_xent(logit, lbl)
                else:
                    xent_loss += b_xent(logit, lbl)
            ''' Discriminator based loss. '''
            loss = xent_loss
            ''' Consensus regularization loss. '''
            reg_loss = result['reg_loss']
            loss += self.args.reg_coef * reg_loss

            ''' Relation specific clustering loss. '''
            rel_clus_loss, rel_learn_loss = result['rel_clus_assignment_loss'], result['rel_clus_learning_loss']
            loss += self.args.c_assign_coef * (rel_clus_loss) + self.args.c_learn_coef * rel_learn_loss
            if self.args.n_cross_nw_coef != 0:
                cross_reg_loss = result['cross_reg_loss']
                loss += cross_reg_loss

            ''' Supervision loss. '''
            if self.args.isSemi:
                sup = result['semi']
                semi_loss = xent(sup[self.idx_train], self.train_lbls)
                loss = loss + self.args.sup_coef * semi_loss

            ''' Minimum loss based on near-perfect reconstruction. '''
            if loss < best:
                best = loss
                cnt_wait = 0
                best_sup = sup
                torch.save(model, 'saved_model/best_{}_{}_{}_{}_{}_{}_{}_{}.pkl'.format(self.args.dataset, self.args.embedder,
                                                                                 self.args.reg_coef, self.args.sup_coef,
                                                                                 self.args.c_assign_coef, self.args.lr,
                                                                                 self.args.l2_coef, self.args.clusters))
                if self.args.full_run:
                    # print(epoch)
                    model.eval()
                    with torch.no_grad():
                        h_list = []
                        for r in range(self.args.nb_graphs):
                            h_list.append(model.H[r].data.detach())
                        u = model.U.data.detach()
                        h = torch.mean(torch.stack(h_list), 0)

                    if self.args.dataset in ['imdbl']:
                        from evaluation.evaluate_ml import evaluate  # evaluation multi-label
                    else:
                        from evaluation.evaluate_mc import evaluate  # evaluation multi-class

                    if epoch == 0:
                        results = evaluate(u, h, h_list, best_sup, self.idx_train, self.idx_val, self.idx_test, self.labels,
                                           self.args.clusters, self.args.dataset, self.args.embedder, self.args.device)
                        best_results = results
                        model.train()
                    else:
                        if (best_count) % 5 == 0:
                            tmp_best = best_results
                            results = evaluate(u, h, h_list, best_sup, self.idx_train, self.idx_val, self.idx_test, self.labels,
                                               self.args.clusters, self.args.dataset, self.args.embedder, self.args.device)
                            best_results = {key: max(value, results[key]) for key, value in tmp_best.items()}
                            model.train()
                    best_count += 1
            else:
                cnt_wait += 1

            if (epoch) % 125 == 0:
                print("-----------------------------------------------------------------------")
                print("Epoch: ", epoch, "\tResult: ", best_results)
                print("-----------------------------------------------------------------------")

            ''' Stopping criteria for train iterations. '''
            if cnt_wait == self.args.patience:
                break

            loss.backward()
            optimiser.step()
            loss.detach()

        model = torch.load('saved_model/best_{}_{}_{}_{}_{}_{}_{}_{}.pkl'.format(self.args.dataset, self.args.embedder,
                                self.args.reg_coef, self.args.sup_coef, self.args.c_assign_coef, self.args.lr, self.args.l2_coef, self.args.clusters))
        model.eval()
        if self.args.dataset in ['imdbl']:
            from evaluation.evaluate_ml import evaluate  # evaluation multi-label
        else:
            from evaluation.evaluate_mc import evaluate # evaluation multi-class

        print("Completed")

        h_list = []
        for r in range(self.args.nb_graphs):
            h_list.append(model.H[r].data.detach())
        u = model.U.data.detach()
        h = torch.mean(torch.stack(h_list), 0)
        results = evaluate(u, h, h_list, best_sup, self.idx_train, self.idx_val,
                           self.idx_test, self.labels,
                           self.args.clusters, self.args.dataset, self.args.embedder, self.args.device)

        if self.args.full_run:
            tmp_best = best_results
            best_results = {key: max(value, results[key]) for key, value in tmp_best.items()}
        else:
            best_results = results

        print(best_results)
        self.writetofile_fn(best_results)