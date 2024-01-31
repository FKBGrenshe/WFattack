import numpy as np
import random
import time
from sklearn.metrics import confusion_matrix, accuracy_score

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from domain_incremental_dataset import *


def getOneHot(targets, n_classes):
    '''
    targets = [2,3,1]
    onehot = torch.eye(6)[targets]
    tensor([[0., 0., 1., 0., 0., 0.],
            [0., 0., 0., 1., 0., 0.],
            [0., 1., 0., 0., 0., 0.]])
    '''
    return torch.eye(n_classes)[targets]


def fillClassMap(data_list, label_list, n_classes):
    class_map = dict.fromkeys(np.arange(0,n_classes))
    for per_class in class_map:
        class_data_idxes = np.where(label_list == per_class)
        class_map[per_class] = data_list[class_data_idxes]
    return class_map


# dict to list
def formatExemplars(exemplars_set):
    exemplars_data = []
    exemplars_label = []
    for label in exemplars_set:
        for item in exemplars_set[label]:
            exemplars_data.append(item)
            exemplars_label.append(label)
    return exemplars_data, exemplars_label


def update_exemplar_set(old_set, new_set):
    for label in new_set:
        for item in new_set[label]:
            old_set[label].append(item)

    return old_set

def get_ACC(prediciton, groundtrueth):
    # total_num = len(groundtrueth)
    ACC = sum(p == l for p, l in zip(prediciton, groundtrueth))/len(groundtrueth)
    return ACC

def matrix(prediction, groundtruth):

    cnf_matrix = confusion_matrix(groundtruth, prediction)
    
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

    return TPR, FPR, F1



def domain_incremental_trainset(X,y,sample_per_class=20):

    unique_classes = np.unique(y)  # 找到所有唯一的类别
    X_train, X_test, y_train, y_test = [], [], [], []
    for cls in unique_classes:
        # 找到属于当前类别的索引
        indices = np.where(y == cls)[0]
        # 从这个类别的索引中选择前(sample_per_class)20个作为训练集，剩余的作为测试集
        train_indices = indices[:sample_per_class]
        test_indices = indices[sample_per_class:]

        X_train.extend(X[train_indices])
        X_test.extend(X[test_indices])
        y_train.extend(y[train_indices])
        y_test.extend(y[test_indices])

    # 转换为 NumPy 数组
    X_train, X_test = np.array(X_train), np.array(X_test)
    y_train, y_test = np.array(y_train), np.array(y_test)

    return X_train, X_test, y_train, y_test



############################ CLASSIFIERS #################################

def NMEClassifier(classifie_data, classifie_data_label,exemplars_set, net, n_classes, device):
    '''find every class mean'''
    means = dict.fromkeys(np.arange(n_classes))
    net.eval()
    # computing means
    for label in range(n_classes):
        per_class_exemplars = np.array(exemplars_set[label])
        current_class_exemplar_dataset = class_dataset(X=per_class_exemplars,Y=label)
        current_class_exemplar_dataloader = DataLoader(dataset=current_class_exemplar_dataset, batch_size=512, shuffle=False, drop_last=False)

        mean = torch.zeros((1, 512), device=device)

        for loader_datas in current_class_exemplar_dataloader:
            with torch.no_grad():
                loader_datas = loader_datas.to(device)
                loader_datas_outputs = net(loader_datas.to(device), features=True)
                for loader_datas_output in loader_datas_outputs:
                    mean += loader_datas_output
        # calculate 'avg' mean
        mean = mean / len(per_class_exemplars)
        # class mean -> optimize: norm mean
        # means[label] = mean
        means[label] = mean / mean.norm()

    classifie_data = np.array(classifie_data)
    classifie_data_dataset = current_dataset(X=classifie_data,Y=classifie_data_label)
    classifie_data_dataloader = DataLoader(dataset=classifie_data_dataset, batch_size=512, shuffle=True, drop_last=False)

    predictions, label_list = [], []
    # NME prediciting
    for loader_datas, loader_datas_labels in classifie_data_dataloader:
        loader_datas = loader_datas.to(device)
        label_list += list(loader_datas_labels)
        with torch.no_grad():
            loader_datas_outputs = net(loader_datas, features=True)
            for loader_datas_output in loader_datas_outputs:
                prediction = None
                min_dist = 99999
                for label in means:
                    dist = torch.dist(means[label], loader_datas_output)
                    if dist < min_dist:
                        min_dist = dist
                        prediction = label
                predictions.append(prediction)

    # Acc = accuracy_score(label_list, predictions)
    # TPR,FPR,F1 = metrix(predictions, label_list)

    return predictions, label_list

def NMEClassifier_Prediction_Proba(classifie_data, classifie_data_label,exemplars_set, net, n_classes, device):
    '''find every class mean'''
    means = dict.fromkeys(np.arange(n_classes))
    net.eval()
    # computing means
    for label in range(n_classes):
        per_class_exemplars = np.array(exemplars_set[label])
        current_class_exemplar_dataset = class_dataset(X=per_class_exemplars,Y=label)
        current_class_exemplar_dataloader = DataLoader(dataset=current_class_exemplar_dataset, batch_size=128, shuffle=False, drop_last=False,pin_memory=True)

        mean = torch.zeros((1, 512), device=device)

        for loader_datas in current_class_exemplar_dataloader:
            with torch.no_grad():
                loader_datas = loader_datas.to(device)
                loader_datas_outputs = net(loader_datas.to(device), features=True)
                for loader_datas_output in loader_datas_outputs:
                    mean += loader_datas_output
        # calculate 'avg' mean
        mean = mean / len(per_class_exemplars)
        # class mean -> optimize: norm mean
        # means[label] = mean
        means[label] = mean / mean.norm()

    classifie_data = np.array(classifie_data)
    classifie_data_dataset = WFDataset(X=classifie_data,Y=classifie_data_label)
    classifie_data_dataloader = DataLoader(dataset=classifie_data_dataset, batch_size=512, shuffle=True, drop_last=False,pin_memory=True)

    predictions = [] ; label_list = []
    prediction_prob = []
    # NME prediciting_Proba
    for loader_datas, loader_datas_labels in classifie_data_dataloader:
        loader_datas = loader_datas.to(device)
        label_list += list(loader_datas_labels)
        with torch.no_grad():
            loader_datas_outputs = net(loader_datas, features=True)
            # 对 每个 测试数据 的 计算相似性度量，并记录
            for loader_datas_output in loader_datas_outputs:
                prediction = None
                min_dist = 99999
                dist_list_per_data = []; cos_metrics_list_per_data = []
                for label in means:
                    cos_metrics = torch.nn.functional.cosine_similarity(means[label], loader_datas_output)
                    cos_metrics_list_per_data.append(cos_metrics)
                #     # 对dist_list_per_data计算softmax
                # proba_list_per_data = torch.softmax(torch.tensor(dist_list_per_data), dim=0)
                predictions.append(torch.argmax(torch.tensor(cos_metrics_list_per_data)))
                prediction_prob.append(cos_metrics_list_per_data)

                    # dist = torch.dist(means[label], loader_datas_output)
                    # if dist < min_dist:
                    #     min_dist = dist
                    #     prediction = label
                

                # predictions.append(prediction)

    '''NME在分类过程中使用每个类别的均值（mean）作为代表，而不是在训练时存储所有实例'''

    return predictions, label_list, prediction_prob

def compute_exemplars_mean(exemplars_set, net, n_classes, device):
    '''find every class mean'''
    means = dict.fromkeys(np.arange(n_classes))
    net.eval()
    # computing means
    for label in range(n_classes):
        per_class_exemplars = np.array(exemplars_set[label])
        current_class_exemplar_dataset = class_dataset(X=per_class_exemplars,Y=label)
        current_class_exemplar_dataloader = DataLoader(dataset=current_class_exemplar_dataset, batch_size=512, shuffle=False, drop_last=False,pin_memory=True)

        mean = torch.zeros((1, 512), device=device)

        for loader_datas in current_class_exemplar_dataloader:
            with torch.no_grad():
                loader_datas = loader_datas.to(device)
                loader_datas_outputs = net(loader_datas.to(device), features=True)
                for loader_datas_output in loader_datas_outputs:
                    mean += loader_datas_output
        # calculate 'avg' mean
        mean = mean / len(per_class_exemplars)
        # class mean -> optimize: norm mean
        # means[label] = mean
        means[label] = mean / mean.norm()

    return means








if __name__ == '__main__':
    pass