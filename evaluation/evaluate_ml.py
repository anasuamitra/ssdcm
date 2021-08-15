import torch
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
from layers.logreg import LogReg
import torch.nn as nn
import numpy as np
np.random.seed(0)
from sklearn.metrics import f1_score, accuracy_score
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, pairwise
from collections import OrderedDict
from scipy.spatial import distance
from sklearn.decomposition import PCA
from sklearn.decomposition import NMF
from sklearn.preprocessing import MinMaxScaler
from fcmeans import FCM
import subprocess
import string
import heapq
import time

def purity_score(y_true, y_pred):
    y_labeled_voted = np.zeros(y_true.shape)
    labels = np.unique(y_true)
    bins = np.concatenate((labels, [np.max(labels) + 1]), axis=0)

    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred == cluster], bins=bins)
        winner = np.argmax(hist)
        y_labeled_voted[y_pred == cluster] = winner
    return accuracy_score(y_true, y_labeled_voted)

def construct_indicator(y_score, y):
    # rank the labels by the scores directly
    num_label = np.sum(y, axis=1, dtype=np.int)
    y_sort = np.fliplr(np.argsort(y_score, axis=1))
    y_pred = np.zeros_like(y, dtype=np.int)
    for i in range(y.shape[0]):
        for j in range(num_label[i]):
            y_pred[i, y_sort[i, j]] = 1
    return y_pred

def evaluate(embeds, cluster_probs, cluster_prob_list, sup, idx_train, idx_val, idx_test, labels, clusters, dataset, algorithm, device, isTest=True):
    results = OrderedDict()
    hid_units = embeds.shape[2]
    nb_classes = labels.shape[2]
    xent = nn.BCEWithLogitsLoss()
    train_embs = embeds[0, idx_train]
    val_embs = embeds[0, idx_val]
    test_embs = embeds[0, idx_test]

    train_lbls = labels[0, idx_train]
    val_lbls = labels[0, idx_val]
    test_lbls = labels[0, idx_test]
    val_lbls = val_lbls.detach().cpu().numpy()
    test_lbls = test_lbls.detach().cpu().numpy()
    test_cluster_probs = cluster_probs[0, idx_test]
    test_cluster_probs = test_cluster_probs.reshape((test_cluster_probs.shape[0], clusters))
    test_cluster_probs = test_cluster_probs.detach().cpu().numpy()
    for r in range(len(cluster_prob_list)):
        tmp_cluster_probs = cluster_prob_list[r][0, idx_test]
        tmp_cluster_probs = tmp_cluster_probs.reshape((tmp_cluster_probs.shape[0], clusters))
        cluster_prob_list[r] = tmp_cluster_probs.detach().cpu().numpy()

    accs = []
    micro_f1s = []
    macro_f1s = []
    macro_f1s_val = [] ##
    if sup is not None:
        for _ in range(1):
            #log = LogReg(hid_units, nb_classes)
            #opt = torch.optim.Adam(log.parameters(), lr=0.01, weight_decay=0.0)
            #log.to(device)

            val_accs = []; test_accs = []
            val_micro_f1s = []; test_micro_f1s = []
            val_macro_f1s = []; test_macro_f1s = []
            for iter_ in range(1):
                # train
                #log.train()
                #opt.zero_grad()

                #logits = log(train_embs)
                #loss = xent(logits, train_lbls)

                #loss.backward()
                #opt.step()

                # val
                logits = sup[idx_val] # log(val_embs)
                preds = construct_indicator(logits.detach().cpu().numpy(), val_lbls)
                val_acc = np.sum(preds == val_lbls) * 1.0 / val_lbls.shape[0]
                val_f1_macro = f1_score(val_lbls, preds, average='macro')
                val_f1_micro = f1_score(val_lbls, preds, average='micro')

                val_accs.append(val_acc.item())
                val_macro_f1s.append(val_f1_macro)
                val_micro_f1s.append(val_f1_micro)

                # test
                logits = sup[idx_test] #log(test_embs)
                preds = construct_indicator(logits.detach().cpu().numpy(), test_lbls)

                test_acc = np.sum(preds == test_lbls) / test_lbls.shape[0]
                test_f1_macro = f1_score(test_lbls, preds, average='macro')
                test_f1_micro = f1_score(test_lbls, preds, average='micro')

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

    results["micro_f1"] = np.mean(micro_f1s)
    results["micro_f1_std"] = np.std(micro_f1s)
    results["macro_f1"] = np.mean(macro_f1s)
    results["macro_f1_std"] = np.std(macro_f1s)

    test_embs = np.array(test_embs.cpu())
    test_lbls = np.array(test_lbls)
    results["nmi"] = run_cmeans(test_embs, test_lbls, clusters, nb_classes, dataset, algorithm)
    results["nmi_h"] = run_cmeans_h(test_cluster_probs, test_lbls, clusters, nb_classes, dataset, algorithm)
    for r in range(len(cluster_prob_list)):
        name = "nmi_h" + str(r)
        results[name] = run_cmeans_h(cluster_prob_list[r], test_lbls, clusters, nb_classes, dataset, algorithm)
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
        tmp = list()
        for i in range(numRows):
            sim = 0
            for j in range(N):
                selected_label = test_lbls[indices[i][j]]
                original_label = test_lbls[i]
                sim += distance.jaccard(selected_label, original_label)
            sim = sim * 1.0 / N
            tmp.append(sim)
        st.append(str(np.round(1.0 - np.mean(tmp),4)))
    return st

def run_cmeans(x, y, k, c, dataset, algorithm):
    # n_labels = c
    # if k > n_labels:
    #     pca = PCA(n_components=n_labels)
    #     x = pca.fit_transform(x)
    # elif k <  n_labels:
    #     scaler = MinMaxScaler()
    #     x = scaler.fit_transform(x)
    #     nmf = NMF(n_components=n_labels, init='random', random_state=0)
    #     x = nmf.fit_transform(x)

    fcm = FCM(n_clusters=c)
    original_labels = y.T
    original_labels_list = list()
    predicted_labels_list = list()

    f = open('results/'+ dataset+'_'+algorithm+'_orig.txt', 'w')
    for j in original_labels:
        tmp = list(np.where(j==1)[0])
        original_labels_list.append(tmp)
        for item in tmp:
            f.write('%d\t' % item)
        f.write('\n')
    f.close()

    NMI_list = []
    for i in range(10):
        fcm.fit(x)
        fcm_centers = fcm.centers
        fcm_labels = fcm.u

        preds = np.array(np.zeros((fcm_labels.shape[0], c)), dtype=np.int)
        for i in range(fcm_labels.shape[0]):
            cluster_indices = heapq.nlargest(len(np.where(y[i] == 1)[0].tolist()), range(len(fcm_labels[i])), fcm_labels[i].take)
            preds[i][cluster_indices] = 1
        predicted_labels = preds.T
        f = open('results/'+ dataset+'_'+algorithm+'_pred.txt', 'w')
        for j in predicted_labels:
            tmp = list(np.where(j == 1)[0])
            predicted_labels_list.append(tmp)
            for item in tmp:
                f.write('%d\t' % item)
            f.write('\n')
        f.close()

        s = './onmi -a'+ ' ' +'results/'+dataset+'_'+algorithm+'_orig.txt' + ' ' +'results/'+dataset+'_'+algorithm+'_pred.txt'
        p = subprocess.Popen(s, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        p.wait()
        time.sleep(5)
        result = p.communicate()[0]
        res = result.split()
        res = [item.decode("utf-8") for item in res]
        # res = [item.translate(str.maketrans('', '', string.punctuation)) for item in res]
        if 'NMImax:' in res:
            val = res[res.index('NMImax:') + 1]
            val = float(val.replace(',', ''))
        NMI_list.append(val)
    s1 = sum(NMI_list) / len(NMI_list)
    # print('\t[Clustering] NMI: {:.4f}'.format(s1))

    return s1

def run_cmeans_h(q, y, k, c, dataset, algorithm):
    val = 0.0
    n_labels = c
    if k > n_labels:
        pca = PCA(n_components=n_labels)
        x = pca.fit_transform(q)
    elif k <  n_labels:
        scaler = MinMaxScaler()
        q = scaler.fit_transform(q)
        nmf = NMF(n_components=n_labels, init='random', random_state=0)
        x = nmf.fit_transform(q)
    else :
        x = q
    # fcm = FCM(n_clusters=n_labels)
    original_labels = y.T
    original_labels_list = list()
    predicted_labels_list = list()

    f = open('results/'+dataset+'_'+algorithm+'_orig.txt', 'w')
    for j in original_labels:
        tmp = list(np.where(j==1)[0])
        original_labels_list.append(tmp)
        for item in tmp:
            f.write('%d\t' % item)
        f.write('\n')
    f.close()

    NMI_list = []
    for i in range(1): # counts can be increased
        preds = np.array(np.zeros((y.shape[0], n_labels)), dtype=np.int)
        for i in range(y.shape[0]):
            cluster_indices = heapq.nlargest(len(np.where(y[i] == 1)[0].tolist()), range(len(x[i])), x[i].take)
            preds[i][cluster_indices] = 1
        predicted_labels = preds.T
        f = open('results/'+dataset+'_'+algorithm+'_pred.txt', 'w')
        for j in predicted_labels:
            tmp = list(np.where(j == 1)[0])
            predicted_labels_list.append(tmp)
            for item in tmp:
                f.write('%d\t' % item)
            f.write('\n')
        f.close()

        s = './onmi -a'+ ' ' + 'results/'+dataset+'_'+algorithm+'_orig.txt' + ' ' + 'results/'+dataset+'_'+algorithm+'_pred.txt'
        p = subprocess.Popen(s, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        p.wait()
        time.sleep(2)
        result = p.communicate()[0]
        res = result.split()
        res = [item.decode("utf-8") for item in res]
        # res = [item.translate(str.maketrans('', '', string.punctuation)) for item in res]
        if 'NMImax:' in res:
            val = res[res.index('NMImax:') + 1]
            val = float(val.replace(',', ''))
        NMI_list.append(val)
    s1 = sum(NMI_list) / len(NMI_list)
    # print('\t[Clustering] NMI: {:.4f}'.format(s1))

    return s1