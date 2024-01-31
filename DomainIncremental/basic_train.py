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
TYPE = 'AWF'
AWF_closeworld_100w_data, AWF_closeworld_100w_label = load_AWF_closeworld_100w()
all_data_list = [AWF_closeworld_100w_data]
all_label_list = [AWF_closeworld_100w_label]
# TYPE = 'WalkieTalkiw'
# TYPE = 'WTFPAD'
# DF_closeworld_WalkieTalkie_data_train, DF_closeworld_WalkieTalkie_data_test, DF_closeworld_WalkieTalkie_label_train, DF_closeworld_WalkieTalkie_label_test = load_DF_closeworld_WTFPAD()
# all_data_list = [DF_closeworld_WalkieTalkie_data_train]
# all_label_list = [DF_closeworld_WalkieTalkie_label_train]
# -------------------------------------------- #
cudnn.benchmark = True
DEVICE = device
BATCH_SIZE = 128
LR = 0.001
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
EPOCHS = 1  # 70
MEMORY = 3690  # Total number of exemplars (10+3*5) * 180 =
HERDING = True  # Wheter to perform herding or random selection for the exemplar set
CLASSIFIER = 'NME'
NUM_CLASSES = 100
params = {
    'EPOCHS': EPOCHS,
    'BATCH_SIZE': BATCH_SIZE,
    'LR': LR,
    'MOMENTUM': MOMENTUM,
    'WEIGHT_DECAY': WEIGHT_DECAY
    }
# -------------------------------------------- #
model = DomainIncrementalModel()
net = DFnet(in_channel=1, num_classes=NUM_CLASSES)
exemplars_set = {}
test_data_per_TASK =[]
test_label_per_TASK =[]
# -------------------------------------------- #
# train and test
ACC_list = []
current_data_test_ACC_list = []
# -------------------------------------------- #

for TASK in range(len(all_data_list)):
    if TASK == 0 :
        number_of_new_exemplars_per_class = 10
    else:
        number_of_new_exemplars_per_class = 3
    # prepare train and test 
    if TYPE == 'WalkieTalkiw':
        print("TYPE: ",TYPE)
        # train_data, test_data, train_label, test_label = DF_closeworld_WalkieTalkie_data_train, DF_closeworld_WalkieTalkie_data_test, DF_closeworld_WalkieTalkie_label_train, DF_closeworld_WalkieTalkie_label_test
    else:
        data = all_data_list[TASK]
        label = all_label_list[TASK]
        train_data, test_data, train_label, test_label = train_test_split(data, label, test_size=0.2, random_state=randomseed,shuffle=True)

    # update model
    net = model.update_Representation(data=train_data,label=train_label,exemplars_set=exemplars_set,net=net,n_classes=NUM_CLASSES, TASK=0)
    new_exemplars = model.build_Exemplarsset(data_list=train_data,label_list=train_label,net=net,number_of_exemplars_per_class=number_of_new_exemplars_per_class,n_classes=NUM_CLASSES,type='build')
    if TASK == 0:
        exemplars_set = new_exemplars
        test_data_per_TASK = test_data
        test_label_per_TASK = test_label
    else:
        exemplars_set = update_exemplar_set(old_set=exemplars_set,new_set=new_exemplars)
        test_data_per_TASK = np.vstack((test_data, test_data_per_TASK))
        # test_data_per_TASK += test_data
        test_label_per_TASK = np.hstack((test_label, test_label_per_TASK))
        # test_label_per_TASK += test_label

    # test 
    # cur_predictions, cur_label_list = NMEClassifier(classifie_data=test_data,classifie_data_label=test_label,exemplars_set=exemplars_set,net=net,n_classes=NUM_CLASSES,device=DEVICE)
    cur_predictions, cur_label_list, prob_list = NMEClassifier_Prediction_Proba(classifie_data=test_data,classifie_data_label=test_label,exemplars_set=exemplars_set,net=net,n_classes=NUM_CLASSES,device=DEVICE)
    
    cur_Acc = accuracy_score(cur_label_list, cur_predictions)
    TPR,FPR,F1 = matrix(groundtruth = cur_label_list, prediction = cur_predictions)
    
    current_data_test_ACC_list.append(cur_Acc)

print(f"test: Acc:{cur_Acc} -- TPR:{TPR.mean()} -- FPR:{FPR.mean()} -- F1:{F1.mean()}")
# print("ACC_list: ",ACC_list)
# print("current_data_test_ACC_list: ",current_data_test_ACC_list)
print("finish!")



