import torch
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
from layers.logreg import LogReg
import torch.nn as nn
import numpy as np
np.random.seed(0)
from sklearn.metrics import f1_score
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, pairwise
from sklearn.decomposition import PCA
from sklearn.decomposition import NMF
from sklearn.preprocessing import MinMaxScaler
from collections import OrderedDict

def evaluate(embeds, cluster_probs, cluster_prob_list, sup, idx_train, idx_val, idx_test, labels,  clusters, dataset, algorithm, device, isTest=True):
    results = OrderedDict()
    hid_units = embeds.shape[2]
    nb_classes = labels.shape[2]
    xent = nn.CrossEntropyLoss()
    train_embs = embeds[0, idx_train]
    val_embs = embeds[0, idx_val]
    test_embs = embeds[0, idx_test]
    test_cluster_probs = cluster_probs[0, idx_test]
    test_cluster_probs = test_cluster_probs.reshape((test_cluster_probs.shape[0], clusters))
    test_cluster_probs = test_cluster_probs.detach().cpu().numpy()
    for r in range(len(cluster_prob_list)):
        tmp_cluster_probs = cluster_prob_list[r][0, idx_test]
        tmp_cluster_probs = tmp_cluster_probs.reshape((tmp_cluster_probs.shape[0], clusters))
        cluster_prob_list[r] = tmp_cluster_probs.detach().cpu().numpy()
    # print(test_cluster_probs.shape) # torch.Size([2950, 3])
    train_lbls = torch.argmax(labels[0, idx_train], dim=1)
    val_lbls = torch.argmax(labels[0, idx_val], dim=1)
    test_lbls = torch.argmax(labels[0, idx_test], dim=1)

    accs = []
    micro_f1s = []
    macro_f1s = []
    macro_f1s_val = [] ##
    if sup is not None:
        for _ in range(1):
            # log = LogReg(hid_units, nb_classes)
            # opt = torch.optim.Adam(log.parameters(), lr=0.01, weight_decay=0.0)
            # log.to(device)

            val_accs = []; test_accs = []
            val_micro_f1s = []; test_micro_f1s = []
            val_macro_f1s = []; test_macro_f1s = []
            for iter_ in range(1):
                # train
                # log.train()
                # opt.zero_grad()
                #
                # logits = log(train_embs)
                # loss = xent(logits, train_lbls)
                #
                # loss.backward()
                # opt.step()

                # val
                # logits = log(val_embs)
                logits = sup[idx_val]
                preds = torch.argmax(logits, dim=1)

                val_acc = torch.sum(preds == val_lbls).float() / val_lbls.shape[0]
                val_f1_macro = f1_score(val_lbls.cpu(), preds.cpu(), average='macro')
                val_f1_micro = f1_score(val_lbls.cpu(), preds.cpu(), average='micro')

                val_accs.append(val_acc.item())
                val_macro_f1s.append(val_f1_macro)
                val_micro_f1s.append(val_f1_micro)

                # test
                logits = sup[idx_test] #log(test_embs)
                preds = torch.argmax(logits, dim=1)

                test_acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
                test_f1_macro = f1_score(test_lbls.cpu(), preds.cpu(), average='macro')
                test_f1_micro = f1_score(test_lbls.cpu(), preds.cpu(), average='micro')

                test_accs.append(test_acc.item())
                test_macro_f1s.append(test_f1_macro)
                test_micro_f1s.append(test_f1_micro)


        max_iter = val_accs.index(max(val_accs))
        accs.append(test_accs[max_iter])

        max_iter = val_macro_f1s.index(max(val_macro_f1s))
        macro_f1s.append(test_macro_f1s[max_iter])
        macro_f1s_val.append(val_macro_f1s[max_iter]) ###

        max_iter = val_micro_f1s.index(max(val_micro_f1s))
        micro_f1s.append(test_micro_f1s[max_iter])

    # if isTest:
    #     print("\t[Classification] Macro-F1: {:.4f} ({:.4f}) | Micro-F1: {:.4f} ({:.4f})".format(np.mean(macro_f1s),
    #                                                                                             np.std(macro_f1s),
    #                                                                                             np.mean(micro_f1s),
    #                                                                                             np.std(micro_f1s)))
    # else:
    #     return np.mean(macro_f1s_val), np.mean(macro_f1s)

    results["micro_f1"] = np.mean(micro_f1s)
    results["micro_f1_std"] = np.std(micro_f1s)
    results["macro_f1"] = np.mean(macro_f1s)
    results["macro_f1_std"] = np.std(macro_f1s)

    test_embs = np.array(test_embs.cpu())
    test_lbls = np.array(test_lbls.cpu())

    results["nmi"] = run_kmeans(test_embs, test_lbls, clusters, nb_classes)
    results["nmi_h"] = run_kmeans_h(test_cluster_probs, test_lbls, clusters, nb_classes)
    for r in range(len(cluster_prob_list)):
        name = "nmi_h" + str(r)
        results[name] = run_kmeans_h(cluster_prob_list[r], test_lbls, clusters, nb_classes)
    st = run_similarity_search(test_embs, test_lbls)
    cols = ["S@5", "S@10", "S@20", "S@50", "S@100"]
    for i in range(len(cols)) :
        results[cols[i]] = st[i]

    return results

def run_similarity_search(test_embs, test_lbls):
    numRows = test_embs.shape[0]

    cos_sim_array = pairwise.cosine_similarity(test_embs) - np.eye(numRows)
    st = []
    for N in [5, 10, 20, 50, 100]:
        indices = np.argsort(cos_sim_array, axis=1)[:, -N:]
        tmp = np.tile(test_lbls, (numRows, 1))
        selected_label = tmp[np.repeat(np.arange(numRows), N), indices.ravel()].reshape(numRows, N)
        original_label = np.repeat(test_lbls, N).reshape(numRows,N)
        st.append(str(np.round(np.mean(np.sum((selected_label == original_label), 1) / N), 4)))

    # st = ','.join(st)
    # print("\t[Similarity] [5,10,20,50,100] : [{}]".format(st))

    return st

def run_kmeans(x, y, k, c):
    estimator = KMeans(n_clusters=c)

    NMI_list = []
    for i in range(10):
        estimator.fit(x)
        y_pred = estimator.predict(x)
        s1 = normalized_mutual_info_score(y, y_pred, average_method='arithmetic')
        NMI_list.append(s1)

    s1 = sum(NMI_list) / len(NMI_list)
    # print('\t[Clustering] NMI: {:.4f}'.format(s1))

    return s1

def run_kmeans_h(q, y, k, c):
    n_labels = c
    y_list = y
    if k == n_labels:
        y_pred = q.argmax(1)
        s1 = normalized_mutual_info_score(y_list, y_pred, average_method='arithmetic')
        # print('\t[Clustering] NMI: {:.4f}'.format(s1))
        return s1
    elif k > n_labels:
        pca = PCA(n_components=n_labels)
        x = pca.fit_transform(q)
    elif k <  n_labels:
        scaler = MinMaxScaler()
        q = scaler.fit_transform(q)
        nmf = NMF(n_components=n_labels, init='random', random_state=0)
        x = nmf.fit_transform(q)
    y_pred_list = x.argmax(1)
    s1 = normalized_mutual_info_score(y_list, y_pred_list, average_method='arithmetic')
    # print('\t[Clustering] NMI: {:.4f}'.format(s1))

    return s1