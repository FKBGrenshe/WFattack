import torch.optim
import torch.nn as nn
import model
from dataset import *
from utils import *
from sklearn.model_selection import train_test_split
torch.backends.cudnn.enable =True
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")

def concept_drift_curent_dataset_train_test():
    # hyperparameters
    NUM_CLASSES = 189
    NB_EPOCH = 5  # 训练epoch
    BATCH_SIZE = 128  # 批大小
    LEARNING_RATE = 0.002  # 学习率
    LENGTH = 5000
    INPUT_SHAPE = (LENGTH, 1)
    # ----------------------------------------------#
    # --------------性能record----------------------#
    # ----------------------------------------------#
    test_acc_current_dataset = []
    test_FPR_current_dataset = []; test_TPR_current_dataset = []
    test_F1_current_dataset = []
    # loading dataset
    AWF_closeword_200w_data, AWF_closeword_200w_label, AWF_concept_3d_data, AWF_concept_3d_label, AWF_concept_10d_data, AWF_concept_10d_label, AWF_concept_2w_data, AWF_concept_2w_label,AWF_concept_4w_data, AWF_concept_4w_label,AWF_concept_6w_data, AWF_concept_6w_label = load_ALL_dataloader()
    TASK0_train_data,TASK0_test_data,TASK0_train_label,TASK0_test_label = train_test_split(AWF_closeword_200w_data,AWF_closeword_200w_label,test_size=0.2,shuffle=True)
    all_data_list = [TASK0_test_data, AWF_concept_3d_data, AWF_concept_10d_data, AWF_concept_2w_data,AWF_concept_4w_data,AWF_concept_6w_data]
    all_label_list = [ TASK0_test_label , AWF_concept_3d_label , AWF_concept_10d_label , AWF_concept_2w_label, AWF_concept_4w_label, AWF_concept_6w_label]
    # -------------------------------------------- #
    train_dataset = AWFDataset(X=TASK0_train_data,Y=TASK0_train_label)
    trian_loader = DataLoader(dataset=train_dataset,shuffle=True,batch_size=BATCH_SIZE)
    # build model
    DFNET = model.DFnet(in_channel=1, num_classes=NUM_CLASSES)
    DFNET.to(device)
    OPTIMIZER = torch.optim.Adamax(DFNET.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1e-08,weight_decay=0.0)
    LOSS = nn.CrossEntropyLoss()
    # -------------------------------------------- #
    # --------------Train stage--------------------#
    # -------------------------------------------- #
    for epoch in range(NB_EPOCH):
        i = 0
        for data, train_label in trian_loader:
            i += 1
            trian_label_onehot = getOneHot(targets = train_label, n_classes = NUM_CLASSES)

            # forward
            OPTIMIZER.zero_grad()  # 梯度清零
            pred = DFNET.forward(data.to(device))

            # backward
            lossvalue = LOSS(pred, trian_label_onehot.to(device))
            lossvalue.backward()  # 反向传播，计算梯度
            OPTIMIZER.step()  # 更新权重

            # 计算每个 epoch 的 batch 的 ACC
            
            ACC = get_ACC(torch.argmax(pred, dim=1).cpu(),train_label)
    # print(f"epoch:{epoch} -- batch:{i} -- train.loss:{lossvalue} -- train.ACC{ACC}")
    # -------------------------------------------- #
    # --------------Test stage--------------------#
    # -------------------------------------------- #
    # print("testing")
    DFNET.eval()
    for TASK in range(len(all_data_list)):
        count = 0
        per_TASK_acc = 0.
        test_data = all_data_list[TASK]
        test_label = all_label_list[TASK]
        test_dataset = AWFDataset(X=test_data,Y=test_label)
        test_loader = DataLoader(dataset=test_dataset,shuffle=True,batch_size=BATCH_SIZE)
        # ---------------- #
        # -----testing---- #
        # ---------------- #
        all_pred = []
        all_label = []
        per_TASK_acc=0.0;per_TASK_tpr=0.0;per_TASK_fpr=0.0;per_TASK_f1=0.0  
        AVG_acc = 0.0;AVG_TPR=0.0;AVG_FPR=0.0;AVG_F1=0.0
        with torch.no_grad():
            count = 0
            for testdata, testdata_label in test_loader:
                count += 1
                pred = DFNET(testdata.to(device))
                all_pred.append(torch.argmax(pred, dim=1).cpu())
                all_label.append(testdata_label)
                ACC = get_ACC(torch.argmax(pred, dim=1).cpu(),testdata_label)
                TPR,FPR,F1 = matrix(torch.argmax(pred, dim=1).cpu(),testdata_label)
                # print(f"TASK:{TASK} -- batch:{count} -- test.ACC{ACC}")
                per_TASK_acc += ACC
                per_TASK_tpr += TPR.mean();per_TASK_fpr += FPR.mean();per_TASK_f1 += F1.mean()
            AVG_acc = per_TASK_acc/count
            AVG_TPR = per_TASK_tpr/count;AVG_FPR = per_TASK_fpr/count;AVG_F1 = per_TASK_f1/count 
        # print(f"TASK{TASK} -- test.ACC:{AVG_acc} -- test.FPR:{AVG_FPR} -- test.TPR:{AVG_TPR} -- test.F1:{AVG_F1}")
        test_acc_current_dataset.append(AVG_acc)
        test_FPR_current_dataset.append(AVG_FPR);test_TPR_current_dataset.append(AVG_TPR)
        test_F1_current_dataset.append(AVG_F1)
        # print(f"TASK{TASK} training and testing finish!")

    # print("all dataset finish!")
    # return ACC, TPR, FPR, F1
    return test_acc_current_dataset,test_FPR_current_dataset,test_TPR_current_dataset,test_F1_current_dataset

def concept_drift_all_dataset_train_test():
    # ----------------------------------------------#
    # --------------------超参数---------------------#
    # ----------------------------------------------#
    train_and_test_rate = 0.8
    NUM_CLASSES = 189
    NB_EPOCH = 5  # 训练epoch
    BATCH_SIZE = 128  # 批大小
    LEARNING_RATE = 0.002  # 学习率
    LENGTH = 5000
    INPUT_SHAPE = (LENGTH, 1)
    DELAY = 1
    # ----------------------------------------------#
    # --------------性能record----------------------#
    # ----------------------------------------------#
    test_acc_current_dataset = []
    test_FPR_current_dataset = []; test_TPR_current_dataset = []
    test_F1_current_dataset = []
    # ----------------------------------------------#
    # --------------load dataset--------------------#
    # ----------------------------------------------#
    AWF_closeword_200w_data, AWF_closeword_200w_label, AWF_concept_3d_data, AWF_concept_3d_label, AWF_concept_10d_data, AWF_concept_10d_label, AWF_concept_2w_data, AWF_concept_2w_label,AWF_concept_4w_data, AWF_concept_4w_label,AWF_concept_6w_data, AWF_concept_6w_label = load_ALL_dataloader()
    TASK0_train_data,TASK0_test_data,TASK0_train_label,TASK0_test_label = train_test_split(AWF_closeword_200w_data,AWF_closeword_200w_label,test_size=0.2,shuffle=True)
    all_data_list = [TASK0_test_data, AWF_concept_3d_data, AWF_concept_10d_data, AWF_concept_2w_data,AWF_concept_4w_data,AWF_concept_6w_data]
    all_label_list = [ TASK0_test_label , AWF_concept_3d_label , AWF_concept_10d_label , AWF_concept_2w_label, AWF_concept_4w_label, AWF_concept_6w_label]
    # build model
    DFNET = model.DFnet(in_channel=1, num_classes=NUM_CLASSES)
    DFNET.to(device)
    OPTIMIZER = torch.optim.Adamax(DFNET.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1e-08,weight_decay=0.0)
    LOSS = nn.CrossEntropyLoss()
    # -------------------------------------------- #
    for TASK in range(len(all_data_list)):
        current_data = all_data_list[TASK]
        current_label = all_label_list[TASK]
        if TASK == 0:
            current_data_train, current_data_test, current_label_trian, current_label_test = train_test_split(current_data, current_label, train_size = train_and_test_rate)
        else:
            current_data_train, current_data_test, current_label_trian, current_label_test = train_test_split(current_data, current_label, train_size = 20*NUM_CLASSES)
        # build train dataset
        train_dataset = AWFDataset(X=current_data_train,Y=current_label_trian)
        trian_loader = DataLoader(dataset=train_dataset,shuffle=True,batch_size=BATCH_SIZE)
        if TASK == 0:
            test_data_per_TASK = current_data_test
            test_label_per_TASK = current_label_test
        else:
            test_data_per_TASK = np.vstack((current_data_test , test_data_per_TASK))
            test_label_per_TASK = np.hstack((current_label_test , test_label_per_TASK))
        # build test dataset
        test_dataset = AWFDataset(X=test_data_per_TASK,Y=test_label_per_TASK)
        test_loader = DataLoader(dataset=test_dataset,shuffle=True,batch_size=BATCH_SIZE)
        # print(f"TASK{TASK} training and testing start")
        # -------------------------------------------- #
        # --------------Train stage--------------------#
        # -------------------------------------------- #
        for epoch in range(NB_EPOCH):
                i = 0
                for data, train_label in trian_loader:
                    i += 1
                    trian_label_onehot = getOneHot(targets = train_label, n_classes = NUM_CLASSES)
                    # forward
                    OPTIMIZER.zero_grad()  # 梯度清零
                    pred = DFNET.forward(data.to(device))
                    # backward
                    lossvalue = LOSS(pred, trian_label_onehot.to(device))
                    lossvalue.backward()  # 反向传播，计算梯度
                    OPTIMIZER.step()  # 更新权重
        # -------------------------------------------- #
        # --------------Test  stage--------------------#
        # -------------------------------------------- #
        print("testing")
        DFNET.eval()
        count = 0
        all_pred = []
        all_label = []
        per_TASK_acc=0.0;per_TASK_tpr=0.0;per_TASK_fpr=0.0;per_TASK_f1=0.0  
        AVG_acc = 0.0;AVG_TPR=0.0;AVG_FPR=0.0;AVG_F1=0.0
        with torch.no_grad():
            count = 0
            for testdata, testdata_label in test_loader:
                count += 1
                pred = DFNET(testdata.to(device))
                all_pred.append(torch.argmax(pred, dim=1).cpu())
                all_label.append(testdata_label)
                ACC = get_ACC(torch.argmax(pred, dim=1).cpu(),testdata_label)
                TPR,FPR,F1 = matrix(torch.argmax(pred, dim=1).cpu(),testdata_label)
                # print(f"TASK:{TASK} -- batch:{count} -- test.ACC{ACC}")
                per_TASK_acc += ACC
                per_TASK_tpr += TPR.mean();per_TASK_fpr += FPR.mean();per_TASK_f1 += F1.mean()
            AVG_acc = per_TASK_acc/count
            AVG_TPR = per_TASK_tpr/count;AVG_FPR = per_TASK_fpr/count;AVG_F1 = per_TASK_f1/count 
        # print(f"TASK{TASK} -- test.ACC:{AVG_acc} -- test.FPR:{AVG_FPR} -- test.TPR:{AVG_TPR} -- test.F1:{AVG_F1}")
        test_acc_current_dataset.append(AVG_acc)
        test_FPR_current_dataset.append(AVG_FPR);test_TPR_current_dataset.append(AVG_TPR)
        test_F1_current_dataset.append(AVG_F1)
        # print(f"TASK{TASK} training and testing finish!")
    print("all dataset finish!")
    # return ACC, TPR, FPR, F1
    return test_acc_current_dataset,test_FPR_current_dataset,test_TPR_current_dataset,test_F1_current_dataset

def concept_drift_collect_delay_train(delay=1):
    # ----------------------------------------------#
    # --------------------超参数---------------------#
    # ----------------------------------------------#
    DELAY = delay
    NUM_CLASSES = 189
    NB_EPOCH = 5  # 训练epoch
    BATCH_SIZE = 128  # 批大小
    LEARNING_RATE = 0.002  # 学习率
    LENGTH = 5000
    INPUT_SHAPE = (LENGTH, 1)
    # ----------------------------------------------#
    # --------------性能record----------------------#
    # ----------------------------------------------#
    test_acc_current_dataset = []
    test_FPR_current_dataset = []; test_TPR_current_dataset = []
    test_F1_current_dataset = []
    # ----------------------------------------------#
    # --------------load dataset--------------------#
    # ----------------------------------------------#
    AWF_closeword_200w_data, AWF_closeword_200w_label, AWF_concept_3d_data, AWF_concept_3d_label, AWF_concept_10d_data, AWF_concept_10d_label, AWF_concept_2w_data, AWF_concept_2w_label,AWF_concept_4w_data, AWF_concept_4w_label,AWF_concept_6w_data, AWF_concept_6w_label = load_ALL_dataloader()
    TASK0_train_data,TASK0_test_data,TASK0_train_label,TASK0_test_label = train_test_split(AWF_closeword_200w_data,AWF_closeword_200w_label,test_size=0.2,shuffle=True)
    all_data_list = [TASK0_test_data, AWF_concept_3d_data, AWF_concept_10d_data, AWF_concept_2w_data,AWF_concept_4w_data,AWF_concept_6w_data]
    all_label_list = [ TASK0_test_label , AWF_concept_3d_label , AWF_concept_10d_label , AWF_concept_2w_label, AWF_concept_4w_label, AWF_concept_6w_label]
    # build model
    DFNET = model.DFnet(in_channel=1, num_classes=NUM_CLASSES)
    DFNET.to(device)
    OPTIMIZER = torch.optim.Adamax(DFNET.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1e-08,weight_decay=0.0)
    LOSS = nn.CrossEntropyLoss()
    # -------------------------------------------- #
    # --------------Train stage--------------------#
    # -------------------------------------------- #
    for TASK in range(len(all_data_list)-DELAY):
        # prepare this task trainning and testing dataset
        current_data = all_data_list[TASK]
        current_label = all_label_list[TASK]
        test_data = all_data_list[TASK+DELAY]
        test_label = all_label_list[TASK+DELAY]
        train_dataset = AWFDataset(X=current_data,Y=current_label)
        trian_loader = DataLoader(dataset=train_dataset,shuffle=True,batch_size=BATCH_SIZE)
        test_dataset = AWFDataset(X=test_data,Y=test_label)
        test_loader = DataLoader(dataset=test_dataset,shuffle=True,batch_size=BATCH_SIZE)
        # print(f"TASK{TASK} training and testing start")
        # -------------------------------------------- #
        # --------------Train stage--------------------#
        # -------------------------------------------- #
        for epoch in range(NB_EPOCH):
                i = 0
                for data, train_label in trian_loader:
                    i += 1
                    trian_label_onehot = getOneHot(targets = train_label, n_classes = NUM_CLASSES)
                    # forward
                    OPTIMIZER.zero_grad()  # 梯度清零
                    pred = DFNET.forward(data.to(device))
                    # backward
                    lossvalue = LOSS(pred, trian_label_onehot.to(device))
                    lossvalue.backward()  # 反向传播，计算梯度
                    OPTIMIZER.step()  # 更新权重
        # -------------------------------------------- #
        # --------------Test  stage--------------------#
        # -------------------------------------------- #
        # print("testing")
        DFNET.eval()
        count = 0
        all_pred = []
        all_label = []
        per_TASK_acc=0.0;per_TASK_tpr=0.0;per_TASK_fpr=0.0;per_TASK_f1=0.0  
        AVG_acc = 0.0;AVG_TPR=0.0;AVG_FPR=0.0;AVG_F1=0.0
        with torch.no_grad():
            count = 0
            for testdata, testdata_label in test_loader:
                count += 1
                pred = DFNET(testdata.to(device))
                all_pred.append(torch.argmax(pred, dim=1).cpu())
                all_label.append(testdata_label)
                ACC = get_ACC(torch.argmax(pred, dim=1).cpu(),testdata_label)
                TPR,FPR,F1 = matrix(torch.argmax(pred, dim=1).cpu(),testdata_label)
                # print(f"TASK:{TASK} -- batch:{count} -- test.ACC{ACC}")
                per_TASK_acc += ACC
                per_TASK_tpr += TPR.mean();per_TASK_fpr += FPR.mean();per_TASK_f1 += F1.mean()
            AVG_acc = per_TASK_acc/count
            AVG_TPR = per_TASK_tpr/count;AVG_FPR = per_TASK_fpr/count;AVG_F1 = per_TASK_f1/count 
        # print(f"TASK{TASK} -- test.ACC:{AVG_acc} -- test.FPR:{AVG_FPR} -- test.TPR:{AVG_TPR} -- test.F1:{AVG_F1}")
        test_acc_current_dataset.append(AVG_acc)
        test_FPR_current_dataset.append(AVG_FPR);test_TPR_current_dataset.append(AVG_TPR)
        test_F1_current_dataset.append(AVG_F1)
        # print(f"TASK{TASK} training and testing finish!")
    # print("all dataset finish!")
    # return ACC, TPR, FPR, F1
    return test_acc_current_dataset,test_FPR_current_dataset,test_TPR_current_dataset,test_F1_current_dataset


def main():
    
    curent_ACC, curent_TPR, curent_FPR, curent_F1 = concept_drift_curent_dataset_train_test()
    print("################################ concept_drift_curent_dataset_train_test ################################")
    print(f"curent_ACC:{curent_ACC} -- curent_TPR:{curent_TPR} -- curent_FPR:{curent_FPR} -- curent_F1:{curent_F1}")
    

    all_ACC, all_TPR, all_FPR, all_F1 = concept_drift_all_dataset_train_test()
    print(f"all_ACC:{all_ACC} -- all_TPR:{all_TPR} -- all_FPR:{all_FPR} -- all_F1:{all_F1}")
    print("################################ concept_drift_all_dataset_train_test ################################")
    
    
    delay_ACC, delay_TPR, delay_FPR, delay_F1 = concept_drift_collect_delay_train(delay=2)
    print(f"delay_ACC:{delay_ACC} -- delay_TPR:{delay_TPR} -- delay_FPR:{delay_FPR} -- delay_F1:{delay_F1}")
    print("################################ concept_drift_collect_delay_train ################################")

if __name__ == "__main__":
    main()