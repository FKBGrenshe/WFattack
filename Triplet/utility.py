import torch as t
import numpy as np
import random
import pandas as pd
from sklearn.utils import column_or_1d
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from dataset import *

def get_matrix(y_test, predicted_labels):
    cnf_matrix = confusion_matrix(y_test, predicted_labels)
    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix) 
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/((TP+FN+1e-8))
    # Specificity or true negative rate
    TNR = TN/(TN+FP+1e-8) 
    # Precision or positive predictive value
    PPV = TP/(TP+FP+1e-8)
    # print(TP)
    # print((TP+FP))
    # Negative predictive value
    NPV = TN/(TN+FN+1e-8)
    # Fall out or false positive rate
    FPR = FP/(FP+TN+1e-8)
    # False negative rate
    FNR = FN/(TP+FN+1e-8)
    # False discovery rate
    FDR = FP/(TP+FP+1e-8)
    # Overall accuracy for each class
    ACC = (TP+TN)/(TP+FP+FN+TN)
    F1 = 2*PPV*TPR/(PPV+TPR+1e-8)

    return TPR,FPR,F1

def setup_seed(seed):
     t.manual_seed(seed)
     t.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)


def build_pos_pairs_for_id(classid, classid_to_ids): # classid --> e.g. 0
    traces = classid_to_ids[classid]

    pos_pairs = [(traces[i], traces[j]) for i in range(len(traces)) for j in range(i+1, len(traces))]
    random.shuffle(pos_pairs)
    return pos_pairs

def build_positive_pairs(class_id_range, classid_to_ids):
    # class_id_range = range(0, num_classes)
    listX1 = []
    listX2 = []
    for class_id in class_id_range:
        pos = build_pos_pairs_for_id(class_id, classid_to_ids)
        # -- pos [(1, 9), (0, 9), (3, 9), (4, 8), (1, 4),...] --> (anchor example, positive example)
        for pair in pos:
            listX1 += [pair[0]] # identity
            listX2 += [pair[1]] # positive example
    perm = np.random.permutation(len(listX1))
    # random.permutation([1,2,3]) --> [2,1,3] just random
    # random.permutation(5) --> [1,0,4,3,2]
    # In this case, we just create the random index
    # Then return pairs of (identity, positive example)
    # that each element in pairs in term of its index is randomly ordered.
    return np.array(listX1)[perm], np.array(listX2)[perm]


# ------------------- Hard Triplet Mining -----------
# Naive way to compute all similarities between all network traces.

def build_similarities(conv, all_imgs):
    batch_size = 900
    all_imgs = t.tensor(all_imgs,dtype=t.float32).cuda()
    conv.eval()
    num_imgs = all_imgs.size(0)
    with t.no_grad():
        embs = []
        for i in range(0, num_imgs, batch_size):
            batch_imgs = all_imgs[i:i+batch_size]
            batch_embs = conv(batch_imgs)
            embs.append(batch_embs)
    embs = t.cat(embs)
    embs = embs.cpu() / np.linalg.norm(embs.cpu(), axis=-1, keepdims=True)
    all_sims = np.dot(embs, embs.T)
    return all_sims

def intersect(a, b):
    return list(set(a) & set(b))

def build_negatives(anc_idxs, pos_idxs, similarities, neg_imgs_idx,id_to_classid,alpha_value, num_retries=50):
    # If no similarities were computed, return a random negative
    if similarities is None:
        return random.sample(neg_imgs_idx,len(anc_idxs))
    final_neg = []
    # for each positive pair
    for (anc_idx, pos_idx) in zip(anc_idxs, pos_idxs):
        anchor_class = id_to_classid[anc_idx]
        #positive similarity
        sim = similarities[anc_idx, pos_idx]
        # find all negatives which are semi(hard)
        possible_ids = np.where((similarities[anc_idx] + alpha_value) > sim)[0]
        possible_ids = intersect(neg_imgs_idx, possible_ids)
        appended = False
        for iteration in range(num_retries):
            if len(possible_ids) == 0:
                break
            idx_neg = random.choice(possible_ids)
            if id_to_classid[idx_neg] != anchor_class:
                final_neg.append(idx_neg)
                appended = True
                break
        if not appended:
            final_neg.append(random.choice(neg_imgs_idx))
    return final_neg

class SemiHardTripletGenerator():
    def __init__(self, Xa_train, Xp_train, batch_size, all_traces, neg_traces_idx, conv):
        self.batch_size = batch_size

        self.traces = all_traces
        self.Xa = Xa_train
        self.Xp = Xp_train
        self.cur_train_index = 0
        self.num_samples = Xa_train.shape[0]
        self.neg_traces_idx = neg_traces_idx
        self.all_anchors = list(set(Xa_train))
        self.mapping_pos = {v: k for k, v in enumerate(self.all_anchors)}
        self.mapping_neg = {k: v for k, v in enumerate(self.neg_traces_idx)}
        if conv:
            self.similarities = build_similarities(conv, self.traces)
        else:
            self.similarities = None

    def next_train(self,id_to_classid,alpha_value):
        while 1:
            self.cur_train_index += self.batch_size
            if self.cur_train_index >= self.num_samples:
                self.cur_train_index = 0

            # fill one batch
            traces_a = self.Xa[self.cur_train_index:self.cur_train_index + self.batch_size]
            traces_p = self.Xp[self.cur_train_index:self.cur_train_index + self.batch_size]
            traces_n = build_negatives(traces_a, traces_p, self.similarities, self.neg_traces_idx,id_to_classid,alpha_value,)

            yield self.traces[traces_a],self.traces[traces_p],self.traces[traces_n]



def create_test_set_AWF_close_world(features_model,shot):
    n_query = 70

    
    # train_name = f'/root/datasets/FSCIL/{dataset}.npz'
    # train_dataset = np.load(train_name,allow_pickle=True)

    train_data ,train_labels = load_AWF_closeworld_100w()
    train_labels = np.array([str(lab) for lab in train_labels])
    unique = pd.Series(column_or_1d(train_labels, warn=True)).unique()
    num_base_class = len(unique)
    print(num_base_class)
    
    dic_train = {}
    dic_test = {}
    for i in range(num_base_class):
        inds_train = np.argwhere(train_labels==unique[i])

        samples_train = inds_train.reshape(-1)

        support = np.array(train_data[samples_train][:shot])
        support = support.reshape(support.shape[0],1,support.shape[1])

        query = np.array(train_data[samples_train][shot:shot+n_query])
        query = query.reshape(query.shape[0],1,query.shape[1])

        support = t.tensor(support,dtype=t.float32).cuda()
        query = t.tensor(query,dtype=t.float32).cuda()
        features_model.eval()
        with t.no_grad():
            dic_train[unique[i]] = np.array([np.array(features_model(support).cpu()).mean(axis=0)])
            dic_test[unique[i]] = np.array(features_model(query).cpu())


    return dic_train, dic_test


def create_test_set_AWF_concept_test(features_model,shot,traindata,trainlabels,testdata,testlabel, NUM_CLASS):
    # n_query = 70

    # train_labels = np.array([str(lab) for lab in train_labels])
    # unique = pd.Series(column_or_1d(train_labels, warn=True)).unique()
    # num_base_class = len(unique)
    # print(num_base_class)
    
    dic_train = {}
    dic_test = {}

    for per_class in range(NUM_CLASS):
        idx_train = np.argwhere(trainlabels==per_class); few_shot_idx_train = idx_train[:shot]
        idx_test = np.argwhere(testlabel==per_class)

        test = np.array(testdata[idx_test])
        few_shot_train = np.array(traindata[few_shot_idx_train])

        # few_shot_train = few_shot_train.reshape(few_shot_train.shape[0],1,few_shot_train.shape[1])
        few_shot_train = t.tensor(few_shot_train,dtype=t.float32).cuda()
        # test = test.reshape(test.shape[0],1,test.shape[1])
        test = t.tensor(test,dtype=t.float32).cuda()

        features_model.eval()
        with t.no_grad():
            dic_train[per_class] = np.array([np.array(features_model(few_shot_train).cpu()).mean(axis=0)])
            dic_test[per_class] = np.array(features_model(test).cpu())

    return dic_train, dic_test


def kNN_accuracy(signature_vector_dict, test_vector_dict,n_shot):
    X_train = []
    y_train = []

    # print "Size of problem :", size_of_problem
    site_labels = list(signature_vector_dict.keys())
    print(len(site_labels))
    random.shuffle(site_labels)
    tested_sites = site_labels[:]
    for s in tested_sites:
        for each_test in range(len(signature_vector_dict[s])):
            X_train.append(signature_vector_dict[s][each_test])
            y_train.append(s)

    X_test = []
    y_test = []
    for s in tested_sites:
        for i in range(len(test_vector_dict[s])):
            X_test.append(test_vector_dict[s][i])
            y_test.append(s)

    knn = KNeighborsClassifier(n_neighbors=n_shot, weights='distance', p=2, metric='cosine', algorithm='brute')
    knn.fit(np.array(X_train),np.array(y_train))
    #joblib.dump(knn, f'./trained_model/knn_model_{dataset}_{n_shot}.pkl')

    predict = knn.predict(X_test)
    acc_knn_top1 = accuracy_score(y_test,predict)
    tpr,fpr,f1 = get_matrix(predict,y_test)
    acc_knn_top1 = float("{0:.15f}".format(round(acc_knn_top1, 6)))


    return acc_knn_top1,tpr.mean(),fpr.mean(),f1.mean()