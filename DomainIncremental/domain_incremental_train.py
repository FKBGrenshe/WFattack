import torch
from torch.utils.data import Subset
from torch.backends import cudnn
# import dataset
from domain_incremental_model import DomainIncrementalModel
from domain_incremental_utils import *
from domain_incremental_dataset import *
from domain_incremental_feature_extractor import DFnet
import numpy as np
# --------------- #
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
# --------------- #
'''
website : 180
task 1:
    train dataset: close wolrd 200w 2500tr
    test dataset: close wolrd 200w 2500tr

    add exemplars per site: 10 --> all 1800
task 2:
    train dataset: conceptDrift 200w 3day
    test dataset: close wolrd 200w 2500tr + conceptDrift 200w 3day

    add exemplars per site: 3 --> all 1800 + 3*180*[1]
...
task n: n < 7
    train dataset: conceptDrift 200w
    test dataset: close + concept ...

    add exemplars per site: 3 --> all 1800 + 3*180[n]
'''
# -------------------------------------------- #
# prepare every task dataset
# number2name = get_labelencoder()
# task_0_train_loader, task_0_test_loader,task_0_train_data, task_0_train_label, task_0_test_data, task_0_test_label = get_AWF_closeword_loader(number2name,batchsize=512, shuffle=False)
# task_1_train_loader, task_1_test_loader,task_1_train_data, task_1_train_label, task_1_test_data, task_1_test_label = get_AWF_Concept_Drift_loader(number2name,batchsize=100,delay='3d' ,shuffle=False)
# task_2_train_loader, task_2_test_loader,task_2_train_data, task_2_train_label, task_2_test_data, task_2_test_label = get_AWF_Concept_Drift_loader(number2name,batchsize=100,delay='10d' ,shuffle=False)
# task_3_train_loader, task_3_test_loader,task_3_train_data, task_3_train_label, task_3_test_data, task_3_test_label = get_AWF_Concept_Drift_loader(number2name,batchsize=100, delay='2w',shuffle=False)
# task_4_train_loader, task_4_test_loader,task_4_train_data, task_4_train_label, task_4_test_data, task_4_test_label = get_AWF_Concept_Drift_loader(number2name,batchsize=100, delay='4w',shuffle=False)
# task_5_train_loader, task_5_test_loader,task_5_train_data, task_5_train_label, task_5_test_data, task_5_test_label = get_AWF_Concept_Drift_loader(number2name,batchsize=100, delay='6w',shuffle=False)
AWF_closeword_200w_data, AWF_closeword_200w_label, AWF_concept_3d_data, AWF_concept_3d_label, AWF_concept_10d_data, AWF_concept_10d_label, AWF_concept_2w_data, AWF_concept_2w_label,AWF_concept_4w_data, AWF_concept_4w_label,AWF_concept_6w_data, AWF_concept_6w_label = load_ALL_dataloader()
all_data_list = [AWF_closeword_200w_data, AWF_concept_3d_data, AWF_concept_10d_data, AWF_concept_2w_data,AWF_concept_4w_data,AWF_concept_6w_data]
all_label_list = [ AWF_closeword_200w_label , AWF_concept_3d_label , AWF_concept_10d_label , AWF_concept_2w_label, AWF_concept_4w_label, AWF_concept_6w_label]
# -------------------------------------------- #
cudnn.benchmark = True
DEVICE = device
BATCH_SIZE = 128
LR = 0.001
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
EPOCHS =5   # 70  5
MEMORY = 3690  # Total number of exemplars (10+3*5) * 180 =
HERDING = True  # Wheter to perform herding or random selection for the exemplar set
CLASSIFIER = 'NME'
NUM_CLASSES = 189
params = {
    'EPOCHS': EPOCHS,
    'BATCH_SIZE': BATCH_SIZE,
    'LR': LR,
    'MOMENTUM': MOMENTUM,
    'WEIGHT_DECAY': WEIGHT_DECAY
    }
# -------------------------------------------- #
model = DomainIncrementalModel(memory=MEMORY, device=DEVICE, params=params)
net = DFnet(in_channel=1, num_classes=NUM_CLASSES)
exemplars_set = {}
test_data_per_TASK =[]
test_label_per_TASK =[]
# -------------------------------------------- #
# train and test
ACC_list = []
current_dataset_ACC_list = []
current_dataset_TPR_list = [];current_dataset_FPR_list = [];current_dataset_F1_list = []
# -------------------------------------------- #

for TASK in range(len(all_data_list)):
    # 清空cuda 显存
    torch.cuda.empty_cache()
    # prepare train and test 
    data = all_data_list[TASK]
    label = all_label_list[TASK]
    if TASK == 0 :
        number_of_new_exemplars_per_class = 5
        train_data, test_data, train_label, test_label = train_test_split(data, label, test_size=0.2, random_state=randomseed,shuffle=True)
    else:
        number_of_new_exemplars_per_class = 3
        train_data, test_data, train_label, test_label = domain_incremental_trainset(data,label,sample_per_class=1)
        # params.update({'EPOCHS': 15})
        EPOCHS = 15
        # params.update({'BATCH_SIZE': 64})
    
    

    # update model
    net = model.learning_process(data=train_data,label=train_label,exemplars_set=exemplars_set,net=net,NUM_CLASSES=NUM_CLASSES,EPOCHS=EPOCHS,BATCH_SIZE=BATCH_SIZE ,TASK=TASK)
    # data_list, label_list, net, number_of_exemplars_per_class, NUM_CLASSES, old_exemplarset=None
    new_exemplars = model.alpha_build_memory(data_list=train_data,label_list=train_label,net=net,number_of_exemplars_per_class=number_of_new_exemplars_per_class,NUM_CLASSES=NUM_CLASSES,old_exemplarset=exemplars_set)
    if TASK == 0:
        exemplars_set = new_exemplars
        test_data_per_TASK = test_data
        test_label_per_TASK = test_label
    else:
        exemplars_set = update_exemplar_set(old_set=exemplars_set,new_set=new_exemplars)
        # test_data_per_TASK = np.vstack((test_data , test_data_per_TASK))
        # test_data_per_TASK += test_data
        # test_label_per_TASK = np.hstack((test_label , test_label_per_TASK))
        # test_label_per_TASK += test_label

    # test
    testing = True
    if testing == True:

        '''all time data test''' # NMEClassifier_Prediction_Proba(classifie_data, classifie_data_label,exemplars_set, net, n_classes, device)
        
        cur_predictions, cur_label_list,prediction_prob = NMEClassifier_Prediction_Proba(classifie_data=test_data,classifie_data_label=test_label,exemplars_set=exemplars_set,net=net,n_classes=NUM_CLASSES,device=DEVICE)
        torch.cuda.empty_cache()
        # ACC_list.append(Acc)

        # '''current data test'''
        # if TASK == 0:
        #     current_data_test_ACC_list.append(Acc)
        # else:
        #     cur_Acc, cur_predictions, cur_label_list = NMEClassifier(classifie_data=test_data,classifie_data_label=test_label,exemplars_set=exemplars_set,net=net,n_classes=NUM_CLASSES,device=DEVICE)
        #     current_data_test_ACC_list.append(cur_Acc)

        # cur_predictions, cur_label_list = NMEClassifier(classifie_data=test_data,classifie_data_label=test_label,exemplars_set=exemplars_set,net=net,n_classes=NUM_CLASSES,device=DEVICE)
        cur_TPR, cur_FPR, cur_F1 = matrix(cur_predictions,cur_label_list)
        cur_Acc = get_ACC(cur_predictions,cur_label_list)
        
        # ACC_list.append(ACC_new)
        current_dataset_ACC_list.append(cur_Acc)
        current_dataset_TPR_list.append(cur_TPR.mean())
        current_dataset_FPR_list.append(cur_FPR.mean())
        current_dataset_F1_list.append(cur_F1.mean())

print("ACC_list: ",ACC_list)
print("current_data_test_ACC_list: ",current_dataset_ACC_list)
print("current_data_test_TPR_list: ",current_dataset_TPR_list)
print("current_data_test_FPR_list: ",current_dataset_FPR_list)
print("current_data_test_F1_list: ",current_dataset_F1_list)
print("finish!")



