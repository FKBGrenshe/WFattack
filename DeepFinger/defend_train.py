import torch.optim
import torch.nn as nn
import model
import dataset
import numpy as np

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")

def ACCuracy(prediciton, groundtrueth):
    total_num = groundtrueth.size(0)

    ACC = prediciton.eq(groundtrueth.cpu()).sum() / total_num

    return ACC

# ----------------------------------------------#
# --------------------超参数---------------------#
# ----------------------------------------------#
NB_EPOCH = 30  # 训练epoch
BATCH_SIZE = 128  # 批大小
NB_CLASSES = 95  # 类别数
LEARNING_RATE = 0.002  # 学习率
LENGTH = 5000
INPUT_SHAPE = (LENGTH, 1)


def validating(net, lossfunc, validloader):
    gpu = 0
    count = 0
    loss_avg = 0.0
    ACC_avg = 0.0
    net.eval()
    with torch.no_grad():
        for i, batch in enumerate(validloader):
            count += 1
            valid_data, valid_label = [_.cuda(gpu, non_blocking=True) for _ in batch]

            pred = net(valid_data)
            loss_avg += lossfunc(pred, valid_label)
            ACC_avg += ACCuracy(torch.argmax(pred, dim=1).cpu(), valid_label)

    return ACC_avg / count, loss_avg / count


def training():
    gpu = 0
    # ----------------------------------------------#
    # -------------------模型设置---------------------#
    # ----------------------------------------------#
    DFNET = model.DFnet(in_channel=1, num_classes=NB_CLASSES).to(device)
    OPTIMIZER = torch.optim.Adamax(DFNET.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1e-08,
                                   weight_decay=0.0)
    LOSS = nn.CrossEntropyLoss()
    # ----------------------------------------------#
    trainloader, validloader, testloader = dataset.get_dataLoader('WTFPAD', BATCH_SIZE)
    # ----------------------------------------------#
    for epoch in range(NB_EPOCH):
        for i, batch in enumerate(trainloader):
            # 异步模式将cpu转为gpu
            data, train_label = [_.cuda(gpu, non_blocking=True) for _ in batch]

            # forward
            OPTIMIZER.zero_grad()  # 梯度清零
            pred = DFNET.forward(data)

            # backward
            lossvalue = LOSS(pred, train_label)
            lossvalue.backward()  # 反向传播，计算梯度
            OPTIMIZER.step()  # 更新权重

            # 计算每个 epoch 的 batch 的 ACC
            ACC = ACCuracy(torch.argmax(pred, dim=1).cpu(), train_label)
            print("\r" f"epoch:{epoch} -- batch:{i} -- train.loss:{lossvalue} -- train.ACC{ACC}",end="")

        valid_ACC_avg, valid_loss_avg = validating(net=DFNET, lossfunc=LOSS, validloader=validloader)
        print("----------------------------------------")
        print(f"epoch:{epoch} -- valid.loss:{valid_loss_avg} -- valid.ACC{valid_ACC_avg}")
        print("----------------------------------------")

    # ----------------------------------------------#
    # -------------------testing--------------------#
    # ----------------------------------------------#
    print("testing...")
    test_ACC_avg, test_loss_avg = validating(net=DFNET, lossfunc=LOSS, validloader=testloader)

def testing():
    model_path = "/root/autodl-tmp/VERSION_ONE/DeepFinger/DF_version/Model_save/modelsave1109.pt"
    DFNET = torch.load(model_path)
    OPTIMIZER = torch.optim.Adamax(DFNET.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1e-08,
                                   weight_decay=0.0)
    LOSS = nn.CrossEntropyLoss()
    trainloader, validloader, testloader = dataset.get_dataLoader('NoDef', BATCH_SIZE)
    valid_ACC_avg, valid_loss_avg = validating(net=DFNET, lossfunc=LOSS, validloader=testloader)
    print(f"test.loss:{valid_loss_avg} -- test.ACC{valid_ACC_avg}")



    # pass

if __name__ == '__main__':
    training()
    # testing()
    pass


