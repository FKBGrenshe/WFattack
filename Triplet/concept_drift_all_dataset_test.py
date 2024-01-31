import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from DF_model_Triplet import DF,triplet_loss
import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import column_or_1d
from sklearn.model_selection import train_test_split
from dataset import *
from utility import *
# ----------------------------------------------#
# --------------------超参数---------------------#
# ----------------------------------------------#
# alpha = 0.1
# batch_size_value = 128
# emb_size = 64
# number_epoch = 30
num_base_class = 189
# num_sample = 20
# print("with parameters, Alpha: %s, Batch_size: %s, Embedded_size: %s, Epoch_num: %s"%(alpha, batch_size_value, emb_size, number_epoch))
# SEED=42
# np.random.seed(SEED)
# t.manual_seed(SEED)
# t.cuda.manual_seed_all(SEED)
# alpha_value = float(alpha)

# # ----------------------------------------------#
# # --------------load data ----------------------#
# # ----------------------------------------------#
# AWF_closewolrd_200w_data, AWF_closewolrd_200w_label = load_AWF_closeworld_200w()
# X_train,X_test, y_train, y_test =train_test_split(AWF_closewolrd_200w_data,AWF_closewolrd_200w_label,train_size=num_sample*num_base_class, random_state=42,shuffle=True,stratify=AWF_closewolrd_200w_label)

AWF_closeword_200w_data, AWF_closeword_200w_label, AWF_concept_3d_data, AWF_concept_3d_label, AWF_concept_10d_data, AWF_concept_10d_label, AWF_concept_2w_data, AWF_concept_2w_label,AWF_concept_4w_data, AWF_concept_4w_label,AWF_concept_6w_data, AWF_concept_6w_label = load_ALL_dataloader()
all_data_list = [AWF_closeword_200w_data, AWF_concept_3d_data, AWF_concept_10d_data, AWF_concept_2w_data,AWF_concept_4w_data,AWF_concept_6w_data]
all_label_list = [ AWF_closeword_200w_label , AWF_concept_3d_label , AWF_concept_10d_label , AWF_concept_2w_label, AWF_concept_4w_label, AWF_concept_6w_label]
# # ----------random choose data----------------- #
# base_data = np.array(X_train)
# all_traces = base_data.reshape(base_data.shape[0],1,base_data.shape[1])
# uni_labels = np.unique(y_train)
# id_to_classid = {k: v for k, v in enumerate(y_train)}
# classid_to_ids = {k: np.argwhere(y_train == k).flatten() for k in np.unique(y_train)}
# print("Load traces with ",all_traces.shape)

# # ----------------------------------------------#
# # --------------build positive pairs------------#
# # ----------------------------------------------#
# Xa_train, Xp_train = build_positive_pairs(uni_labels, classid_to_ids)

# # Gather the ids of all network traces that are used for training
# # This just union of two sets set(A) | set(B)
# all_traces_train_idx = list(set(Xa_train) | set(Xp_train))
# print("X_train Anchor: ", Xa_train.shape)
# print("X_train Positive: ", Xp_train.shape)


# batch_size = batch_size_value
# setup_seed(42)

# feature_model = DF(emb_size=emb_size)
# optimizer = t.optim.SGD(feature_model.parameters(),lr=0.001,weight_decay=1e-6,momentum=0.9,nesterov=True)
# cos_sim = nn.CosineSimilarity()
# criterion = triplet_loss(cos_sim,alpha)
# t.cuda.set_device(0)
# feature_model.cuda(0)
# criterion = criterion.cuda(0)
# gen_hard = SemiHardTripletGenerator(Xa_train, Xp_train, batch_size, all_traces, all_traces_train_idx, None).next_train(id_to_classid,alpha_value)
# # At first epoch we don't generate hard triplets
# # ----------------------------------------------#
# # --------------trainning stage-----------------#
# # ----------------------------------------------#
# nb_epochs = number_epoch
# for epoch in tqdm(range(nb_epochs),desc='Epoch'):
#     if epoch == 15:
#         torch.save(feature_model.state_dict(), '/root/autodl-tmp/VERSION_ONE/Triplet/AWF_version/luo_version/save_luo_model/luo_awf_15epoch.pt')
#     for i in range(Xa_train.shape[0] // batch_size+1): 
#         t.cuda.empty_cache()
#         anchor,positive,negative = next(gen_hard)
#         anchor = t.tensor(anchor,dtype=t.float32).cuda()
#         positive = t.tensor(positive,dtype=t.float32).cuda()
#         negative = t.tensor(negative,dtype=t.float32).cuda()
#         feature_model.train()
#         optimizer.zero_grad() 
#         anchor_feature = feature_model(anchor) 
#         positive_feature = feature_model(positive) 
#         negative_feature = feature_model(negative) 

#         loss = criterion(anchor_feature,positive_feature,negative_feature)
#         print(f"Epoch: {epoch} -- batch: {i} -- loss: {loss}", end='\r')
#         loss.backward()
#         optimizer.step()
#     print(loss)
#     # t.save(feature_model.state_dict(), '/root/autodl-tmp/VERSION_ONE/Triplet/AWF_version/luo_version/save_luo_model/luo_awf.pt')
#     gen_hard = SemiHardTripletGenerator(Xa_train, Xp_train, batch_size, all_traces, all_traces_train_idx, feature_model).next_train(id_to_classid,alpha_value)
# # t.save(feature_model.state_dict(), '/root/autodl-tmp/VERSION_ONE/Triplet/AWF_version/luo_version/save_luo_model/luo_awf_30epoch.pt')
'''从这里'''    

# ----------------------------------------------#
# --------------testing stage-------------------#
# ----------------------------------------------#

# ----------------------------------------------#
# --------------load dataset--------------------#
# ----------------------------------------------#

model_path ='/root/autodl-tmp/VERSION_ONE/Triplet/AWF_version/luo_version/save_luo_model/luo_awf_30epoch.pt'
checkpoint = t.load(model_path)
feature_model = DF(64)
feature_model.cuda()
feature_model.load_state_dict(checkpoint)
n_shot_list = [5,10,15,20]
delay = 0

for TASK in range(len(all_data_list) - delay):
    # if TASK == 0:
    few_shot_train_data = all_data_list[TASK]
    few_shot_train_label = all_label_list[TASK]
    test_data = all_data_list[TASK + delay]
    test_label = all_label_list[TASK + delay]
    if TASK == 0:
        all_test_data = test_data
        all_test_label = test_label
    else:
        all_test_data = np.vstack((test_data,all_test_data))
        all_test_label = np.hstack((test_label,all_test_label))
    print("TASK:", TASK)
    for n_shot in n_shot_list:
        acc_list_Top1 = []
        ''''''

        signature_vector_dict, test_vector_dict = create_test_set_AWF_concept_test(feature_model,n_shot,few_shot_train_data,few_shot_train_label,all_test_data,all_test_label,num_base_class)
        # '''这里没有修改完，如何将testdelay改出来，这里没有传入testset'''
        # signature_vector_dict, test_vector_dict = create_test_set_AWF_disjoint(features_model=feature_model,
        #                                                                        dataset=dataset,shot = n_shot)
        # ''''''
        # signature_vector_dict, test_vector_dict = create_test_set_AWF_open(features_model=feature_model,
        #                                                                         dataset='tor_100w_2500tr',shot = n_shot,size = 9000)
        # Measure the performance (accuracy)
        acc_knn_top1,tpr,fpr,f1 = kNN_accuracy(signature_vector_dict, test_vector_dict,n_shot=n_shot)
        acc_list_Top1.append(float("{0:.15f}".format(round(acc_knn_top1, 5))))
        print("N_shot:", n_shot)
        print(str(acc_list_Top1).strip('[]'),tpr,fpr,f1)