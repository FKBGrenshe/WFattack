import torch
import dataset
import model
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import utils

torch.backends.cudnn.enable =True
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")
# ----------------------------------------------#
# --------------------超参数---------------------#
# ----------------------------------------------#
TYPE = 'test'
NUM_CLASSES = 95
NB_EPOCH = 30  # 训练epoch
BATCH_SIZE = 128  # 批大小
LEARNING_RATE = 0.002  # 学习率
LENGTH = 5000
INPUT_SHAPE = (LENGTH, 1)

train_and_test_rate = 0.8
# ----------------------------------------------#
# --------------性能record----------------------#
# ----------------------------------------------#
train_epoch_loss = []
train_epoch_acc = []
test_acc = []
test_FPR = []; test_TPR = []
# ----------------------------------------------#

def AWF_closeword_100w_train_test():
    '''
    实验1: 基准测试
        使用AWF_closeword_100w数据集进行训练和测试
        1) 训练集: AWF_closeword_100w
        2) 测试集: AWF_closeword_100w
        3) 模型: DeepFinger\DomainIncremental\Triplet\Resnet18\Basicnet
        4) 性能度量
            a) 训练集: loss, acc
            b) 测试集: acc,TPR,FPR
    '''
    # load dataset
    AWF_closeword_100w_data, AWF_closeword_100w_label = dataset.load_AWF_closeworld_200w()
    train_data, test_data, train_label, test_label = train_test_split(AWF_closeword_100w_data, AWF_closeword_100w_label, train_size = train_and_test_rate)
        # build train dataset
    train_dataset = dataset.AWFDataset(X=train_data,Y=train_label)
    trian_loader = DataLoader(dataset=train_dataset,shuffle=True,batch_size=BATCH_SIZE)
        # build test dataset
    test_dataset = dataset.AWFDataset(X=test_data,Y=test_label)
    test_loader = DataLoader(dataset=test_dataset,shuffle=True,batch_size=BATCH_SIZE)
    # build model
    DFNET = model.DFnet(in_channel=1, num_classes=NUM_CLASSES)
    DFNET.to(device)
    OPTIMIZER = torch.optim.Adamax(DFNET.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1e-08,weight_decay=0.0)
    LOSS = torch.nn.CrossEntropyLoss()
    # train and test
        # train
    for epoch in range(NB_EPOCH):
        i = 0
        for data, train_label in trian_loader:
            i += 1
            trian_label_onehot = utils.getOneHot(targets = train_label, n_classes = NUM_CLASSES)

            # forward
            OPTIMIZER.zero_grad()  # 梯度清零
            pred = DFNET.forward(data.to(device))

            # backward
            lossvalue = LOSS(pred, trian_label_onehot.to(device))
            lossvalue.backward()  # 反向传播，计算梯度
            OPTIMIZER.step()  # 更新权重

            # 计算每个 epoch 的 batch 的 ACC
            
            ACC = utils.get_ACC(torch.argmax(pred, dim=1).cpu(),train_label)
        print(f"epoch:{epoch} -- batch:{i} -- train.loss:{lossvalue} -- train.ACC{ACC}")
        # saving model
    # torch.save(DFNET.state_dict(), './model/DFNET_AWF_closeword_100w.pkl')
        # test
    print("testing")
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
                ACC = utils.get_ACC(torch.argmax(pred, dim=1).cpu(),testdata_label)
                TPR,FPR,F1 = utils.matrix(torch.argmax(pred, dim=1).cpu(),testdata_label)
                # print(f"batch:{count} -- test.ACC{ACC}")
                per_TASK_acc += ACC
                per_TASK_tpr += TPR.mean();per_TASK_fpr += FPR.mean();per_TASK_f1 += F1.mean()
            AVG_acc = per_TASK_acc/count
            AVG_TPR = per_TASK_tpr/count;AVG_FPR = per_TASK_fpr/count;AVG_F1 = per_TASK_f1/count
            
            print(f"test.ACC:{AVG_acc} -- test.FPR:{AVG_FPR} -- test.TPR:{AVG_TPR} -- test.F1:{AVG_F1}")
    # test_acc, test_FPR, test_TPR = test_model(DFNET, test_loader, test_label)
    
def DF_closeword_WalkieTalkie_train_test():
    '''
    超参数设置  -- 需要将NUM_CLASSES = 100

    实验1: 基准测试
        使用DF_closeword_WalkieTalkie数据集进行训练和测试
        1) 训练集: DF_closeword_WalkieTalkie
        2) 测试集: DF_closeword_WalkieTalkie
        3) 模型: DeepFinger\DomainIncremental\Triplet\Resnet18\Basicnet
        4) 性能度量
            a) 训练集: loss, acc
            b) 测试集: acc,TPR,FPR
    '''
    # load dataset
    train_data,valid_data, test_data, train_label,valid_label, test_label = dataset.load_DF_closeworld_WalkieTalkie()
    # train_data, test_data, train_label, test_label = train_test_split(AWF_closeword_100w_data, AWF_closeword_100w_label, train_size = train_and_test_rate)
        # build train dataset
    train_dataset = dataset.AWFDataset(X=train_data,Y=train_label)
    trian_loader = DataLoader(dataset=train_dataset,shuffle=True,batch_size=BATCH_SIZE)
        # build test dataset
    test_dataset = dataset.AWFDataset(X=test_data,Y=test_label)
    test_loader = DataLoader(dataset=test_dataset,shuffle=True,batch_size=BATCH_SIZE)
    # build model
    DFNET = model.DFnet(in_channel=1, num_classes=NUM_CLASSES)
    DFNET.to(device)
    OPTIMIZER = torch.optim.Adamax(DFNET.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1e-08,weight_decay=0.0)
    LOSS = torch.nn.CrossEntropyLoss()
    # train and test
        # train
    for epoch in range(NB_EPOCH):
        i = 0
        for data, train_label in trian_loader:
            i += 1
            trian_label_onehot = utils.getOneHot(targets = train_label, n_classes = NUM_CLASSES)

            # forward
            OPTIMIZER.zero_grad()  # 梯度清零
            pred = DFNET.forward(data.to(device))

            # backward
            lossvalue = LOSS(pred, trian_label_onehot.to(device))
            lossvalue.backward()  # 反向传播，计算梯度
            OPTIMIZER.step()  # 更新权重

            # 计算每个 epoch 的 batch 的 ACC
            
            ACC = utils.get_ACC(torch.argmax(pred, dim=1).cpu(),train_label)
        print(f"epoch:{epoch} -- batch:{i} -- train.loss:{lossvalue} -- train.ACC{ACC}")
        # saving model
    # torch.save(DFNET.state_dict(), './model/DFNET_AWF_closeword_100w.pkl')
        # test
    print("testing")
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
                ACC = utils.get_ACC(torch.argmax(pred, dim=1).cpu(),testdata_label)
                TPR,FPR,F1 = utils.matrix(torch.argmax(pred, dim=1).cpu(),testdata_label)
                # print(f"batch:{count} -- test.ACC{ACC}")
                per_TASK_acc += ACC
                per_TASK_tpr += TPR.mean();per_TASK_fpr += FPR.mean();per_TASK_f1 += F1.mean()
            AVG_acc = per_TASK_acc/count
            AVG_TPR = per_TASK_tpr/count;AVG_FPR = per_TASK_fpr/count;AVG_F1 = per_TASK_f1/count
            
            print(f"test.ACC:{AVG_acc} -- test.FPR:{AVG_FPR} -- test.TPR:{AVG_TPR} -- test.F1:{AVG_F1}")
    # test_acc, test_FPR, test_TPR = test_model(DFNET, test_loader, test_label)
            
def DF_closeword_WTFPAD_train_test():
    '''
    超参数设置  -- 需要将NUM_CLASSES = 95\ EPOCH = 30

    实验1: 基准测试
        使用DF_closeword_WalkieTalkie数据集进行训练和测试
        1) 训练集: DF_closeword_WalkieTalkie
        2) 测试集: DF_closeword_WalkieTalkie
        3) 模型: DeepFinger\DomainIncremental\Triplet\Resnet18\Basicnet
        4) 性能度量
            a) 训练集: loss, acc
            b) 测试集: acc,TPR,FPR
    '''
    # load dataset
    train_data,valid_data, test_data, train_label,valid_label, test_label = dataset.load_DF_closeworld_WTFPAD()
    # train_data, test_data, train_label, test_label = train_test_split(AWF_closeword_100w_data, AWF_closeword_100w_label, train_size = train_and_test_rate)
        # build train dataset
    train_dataset = dataset.AWFDataset(X=train_data,Y=train_label)
    trian_loader = DataLoader(dataset=train_dataset,shuffle=True,batch_size=BATCH_SIZE)
        # build test dataset
    test_dataset = dataset.AWFDataset(X=test_data,Y=test_label)
    test_loader = DataLoader(dataset=test_dataset,shuffle=True,batch_size=BATCH_SIZE)
    # build model
    DFNET = model.DFnet(in_channel=1, num_classes=NUM_CLASSES)
    DFNET.to(device)
    OPTIMIZER = torch.optim.Adamax(DFNET.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1e-08,weight_decay=0.0)
    LOSS = torch.nn.CrossEntropyLoss()
    # train and test
        # train
    for epoch in range(NB_EPOCH):
        i = 0
        for data, train_label in trian_loader:
            i += 1
            trian_label_onehot = utils.getOneHot(targets = train_label, n_classes = NUM_CLASSES)

            # forward
            OPTIMIZER.zero_grad()  # 梯度清零
            pred = DFNET.forward(data.to(device))

            # backward
            lossvalue = LOSS(pred, trian_label_onehot.to(device))
            lossvalue.backward()  # 反向传播，计算梯度
            OPTIMIZER.step()  # 更新权重

            # 计算每个 epoch 的 batch 的 ACC
            
            ACC = utils.get_ACC(torch.argmax(pred, dim=1).cpu(),train_label)
        print(f"epoch:{epoch} -- batch:{i} -- train.loss:{lossvalue} -- train.ACC{ACC}")
        # saving model
    # torch.save(DFNET.state_dict(), './model/DFNET_AWF_closeword_100w.pkl')
        # test
    print("testing")
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
                ACC = utils.get_ACC(torch.argmax(pred, dim=1).cpu(),testdata_label)
                TPR,FPR,F1 = utils.matrix(torch.argmax(pred, dim=1).cpu(),testdata_label)
                # print(f"batch:{count} -- test.ACC{ACC}")
                per_TASK_acc += ACC
                per_TASK_tpr += TPR.mean();per_TASK_fpr += FPR.mean();per_TASK_f1 += F1.mean()
            AVG_acc = per_TASK_acc/count
            AVG_TPR = per_TASK_tpr/count;AVG_FPR = per_TASK_fpr/count;AVG_F1 = per_TASK_f1/count
            
            print(f"test.ACC:{AVG_acc} -- test.FPR:{AVG_FPR} -- test.TPR:{AVG_TPR} -- test.F1:{AVG_F1}")
    # test_acc, test_FPR, test_TPR = test_model(DFNET, test_loader, test_label)

def DF_closeword_WTFPAD_train_test_NOTonehot():
    '''
    超参数设置  -- 需要将NUM_CLASSES = 95\ EPOCH = 30

    实验1: 基准测试
        使用DF_closeword_WalkieTalkie数据集进行训练和测试
        1) 训练集: DF_closeword_WalkieTalkie
        2) 测试集: DF_closeword_WalkieTalkie
        3) 模型: DeepFinger\DomainIncremental\Triplet\Resnet18\Basicnet
        4) 性能度量
            a) 训练集: loss, acc
            b) 测试集: acc,TPR,FPR
    '''
    # load dataset
    # train_data,valid_data, test_data, train_label,valid_label, test_label = basic_dataset.load_DF_closeworld_WTFPAD()
    X_train, X_valid, X_test, Y_train, Y_valid, Y_test = dataset.load_DF_closeworld_WTFPAD()
    # train_data, test_data, train_label, test_label = train_test_split(AWF_closeword_100w_data, AWF_closeword_100w_label, train_size = train_and_test_rate)
        # build train dataset
    train_dataset = dataset.AWFDataset(X=X_train,Y=Y_train)
    trian_loader = DataLoader(dataset=train_dataset,shuffle=True,batch_size=BATCH_SIZE)
        # build valid dataset
    valid_dataset = dataset.AWFDataset(X=X_valid,Y=Y_valid)
    valid_loader = DataLoader(dataset=valid_dataset,shuffle=True,batch_size=BATCH_SIZE)
        # build test dataset
    test_dataset = dataset.AWFDataset(X=X_test,Y=Y_test)
    test_loader = DataLoader(dataset=test_dataset,shuffle=True,batch_size=BATCH_SIZE)
    # build model
    DFNET = model.DFnet(in_channel=1, num_classes=NUM_CLASSES)
    DFNET.to(device)
    OPTIMIZER = torch.optim.Adamax(DFNET.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1e-08,weight_decay=0.0)
    LOSS = torch.nn.CrossEntropyLoss()
    # train and test
        # train
    for epoch in range(NB_EPOCH):
        i = 0
        for data, train_label in trian_loader:
            i += 1
            # trian_label_onehot = basic_utils.getOneHot(targets = train_label, n_classes = NUM_CLASSES)
            # forward
            OPTIMIZER.zero_grad()  # 梯度清零
            pred = DFNET.forward(data.to(device))
            # backward
            lossvalue = LOSS(pred, train_label.to(device))
            lossvalue.backward()  # 反向传播，计算梯度
            OPTIMIZER.step()  # 更新权重
            # 计算每个 epoch 的 batch 的 ACC
            ACC = utils.get_ACC(torch.argmax(pred, dim=1).cpu(),train_label)
        print(f"epoch:{epoch} -- batch:{i} -- train.loss:{lossvalue} -- train.ACC{ACC}")
        print("evaluating...")
        all_pred = []
        all_label = []
        per_TASK_acc=0.0;per_TASK_tpr=0.0;per_TASK_fpr=0.0;per_TASK_f1=0.0  
        AVG_acc = 0.0;AVG_TPR=0.0;AVG_FPR=0.0;AVG_F1=0.0
        with torch.no_grad():
                count = 0
                for valid_data, valid_label in valid_loader:
                    count += 1
                    pred = DFNET(valid_data.to(device))
                    all_pred.append(torch.argmax(pred, dim=1).cpu())
                    all_label.append(valid_label)
                    ACC = utils.get_ACC(torch.argmax(pred, dim=1).cpu(),valid_label)
                    TPR,FPR,F1 = utils.matrix(torch.argmax(pred, dim=1).cpu(),valid_label)
                    # print(f"batch:{count} -- test.ACC{ACC}")
                    per_TASK_acc += ACC
                    per_TASK_tpr += TPR.mean();per_TASK_fpr += FPR.mean();per_TASK_f1 += F1.mean()
                AVG_acc = per_TASK_acc/count
                AVG_TPR = per_TASK_tpr/count;AVG_FPR = per_TASK_fpr/count;AVG_F1 = per_TASK_f1/count
                
                print(f"test.ACC:{AVG_acc} -- test.FPR:{AVG_FPR} -- test.TPR:{AVG_TPR} -- test.F1:{AVG_F1}")

    print("testing...")
    all_pred = []
    all_label = []
    per_TASK_acc=0.0;per_TASK_tpr=0.0;per_TASK_fpr=0.0;per_TASK_f1=0.0  
    AVG_acc = 0.0;AVG_TPR=0.0;AVG_FPR=0.0;AVG_F1=0.0
    with torch.no_grad():
            count = 0
            for test_data, test_label in test_loader:
                count += 1
                pred = DFNET(valid_data.to(device))
                all_pred.append(torch.argmax(pred, dim=1).cpu())
                all_label.append(test_data)
                ACC = utils.get_ACC(torch.argmax(pred, dim=1).cpu(),test_label)
                TPR,FPR,F1 = utils.matrix(torch.argmax(pred, dim=1).cpu(),test_label)
                # print(f"batch:{count} -- test.ACC{ACC}")
                per_TASK_acc += ACC
                per_TASK_tpr += TPR.mean();per_TASK_fpr += FPR.mean();per_TASK_f1 += F1.mean()
            AVG_acc = per_TASK_acc/count
            AVG_TPR = per_TASK_tpr/count;AVG_FPR = per_TASK_fpr/count;AVG_F1 = per_TASK_f1/count
            
            print(f"test.ACC:{AVG_acc} -- test.FPR:{AVG_FPR} -- test.TPR:{AVG_TPR} -- test.F1:{AVG_F1}")

def load_model_test():
     # load dataset
    AWF_closeword_100w_data, AWF_closeword_100w_label = dataset.load_AWF_closeworld_200w()
    train_data, test_data, train_label, test_label = train_test_split(AWF_closeword_100w_data, AWF_closeword_100w_label, train_size = train_and_test_rate)
        # build train dataset
    train_dataset = dataset.AWFDataset(X=train_data,Y=train_label)
    trian_loader = DataLoader(dataset=train_dataset,shuffle=True,batch_size=BATCH_SIZE)
        # build test dataset
    test_dataset = dataset.AWFDataset(X=test_data,Y=test_label)
    test_loader = DataLoader(dataset=test_dataset,shuffle=True,batch_size=BATCH_SIZE)

    # build model
    DFNET = torch.load('/root/autodl-tmp/VERSION_TWO/AWF_version/DF/save_model/DF_AWF_closeworld_100w_3epoch.pt')
    DFNET.to(device)
    OPTIMIZER = torch.optim.Adamax(DFNET.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1e-08,weight_decay=0.0)
    LOSS = torch.nn.CrossEntropyLoss()

    print("testing")
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
                ACC = utils.get_ACC(torch.argmax(pred, dim=1).cpu(),testdata_label)
                TPR,FPR,F1 = utils.matrix(torch.argmax(pred, dim=1).cpu(),testdata_label)
                # print(f"batch:{count} -- test.ACC{ACC}")
                per_TASK_acc += ACC
                per_TASK_tpr += TPR.mean();per_TASK_fpr += FPR.mean();per_TASK_f1 += F1.mean()
            AVG_acc = per_TASK_acc/count
            AVG_TPR = per_TASK_tpr/count;AVG_FPR = per_TASK_fpr/count;AVG_F1 = per_TASK_f1/count
            
            print(f"test.ACC:{AVG_acc} -- test.FPR:{AVG_FPR} -- test.TPR:{AVG_TPR} -- test.F1:{AVG_F1}")


if __name__ == "__main__":
    # AWF_closeword_100w_train_test()
    # DF_closeword_WalkieTalkie_train_test()
    DF_closeword_WTFPAD_train_test_NOTonehot()
    # load_model_test()
