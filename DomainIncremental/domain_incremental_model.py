from copy import deepcopy
import torch
import torch.nn as nn
from domain_incremental_utils import *
from domain_incremental_dataset import *

# --------------- #
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")


# --------------- #

class DomainIncrementalModel():
    def __init__(self, device_=device):
        self.device = device_

    def learning_process(self, data, label, exemplars_set, net, NUM_CLASSES, EPOCHS, BATCH_SIZE, TASK):
        '''
        BATCH_SIZE -- 指示每个batch的大小
        EPOCHS -- 指示训练轮数
        NUM_CLASSES -- 指示类别总数
        TASK -- 指示第几轮任务
        '''
        # 1st: 模型设置
        net.to(device)
        # 损失函数
        criterion = nn.CrossEntropyLoss()
        # 优化器
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
        # 2nd: 数据集设置
        # combine exemplars and new data
        if len(exemplars_set) != 0:
            exemplars_data, exemplars_label = formatExemplars(exemplars_set=exemplars_set)
            data = np.concatenate((data, exemplars_data), axis=0)
            label = np.concatenate((label, exemplars_label), axis=0)
        cur_dataset = WFDataset(data, label)
        cur_dataloader = DataLoader(dataset=cur_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False,
                                    pin_memory=True)
        # 3rd: 训练
        net.train()
        for epoch in range(EPOCHS):
            running_loss = 0.0
            for loader_data, loader_label in cur_dataloader:
                # 独热编码
                loader_label_one_hot = getOneHot(loader_label, NUM_CLASSES).to(device)
                # training
                optimizer.zero_grad()
                loader_data_output = net(loader_data.to(device))
                total_loss = criterion(loader_data_output, loader_label_one_hot)
                # backward
                total_loss.backward()
                optimizer.step()
                # caculate total running_loss for per batch
                running_loss += total_loss.item() * loader_data.size(0)  # total_loss * (batchsize)128
            # Train loss of current epoch
            train_loss = running_loss / len(data)
            print(f"epoch:{epoch} -- trainloss:{train_loss}")

        return net

    def build_memory(self, data_list, label_list, net, number_of_exemplars_per_class, NUM_CLASSES):
        torch.cuda.empty_cache()
        # to find the exemplars
        new_exemplar_set = dict.fromkeys(np.arange(0, NUM_CLASSES))
        for label in new_exemplar_set:
            new_exemplar_set[label] = []
        # build a dict that form like {classes : data1, data2, ...}
        class_map = fillClassMap(data_list, label_list, NUM_CLASSES)
        # get and save net outputs for each class
        net.eval()
        for label in class_map:
            print('\r'f'building exemplars for class {label}', end='', flush=True)
            if len(class_map[label]) < number_of_exemplars_per_class:
                for data in class_map[label]:
                    new_exemplar_set[label].append(data)
            else:
                class_outputs = []
                mean = 0
                # calculate class mean
                with torch.no_grad():
                    class_data_dataset = class_dataset(class_map[label], label)
                    class_data_dataloader = DataLoader(dataset=class_data_dataset, batch_size=128, shuffle=False,
                                                       drop_last=False, pin_memory=True)
                    for class_datas in class_data_dataloader:
                        class_datas_outputs = net(class_datas.to(device), features=True)
                        for output in class_datas_outputs:
                            class_outputs.append(output.unsqueeze(0))
                            mean += output
                        final_outputs = torch.cat(class_outputs, dim=0)
                    # get mean per class
                    mean = mean / len(class_map[label])
                    w_t = mean
                    for i in range(number_of_exemplars_per_class):
                        similarity_list = F.cosine_similarity(w_t, final_outputs)
                        max_similarity_idx = torch.argmax(similarity_list)
                        w_t = w_t + mean - class_outputs[max_similarity_idx]
                        class_outputs.pop(max_similarity_idx)
                        final_outputs = torch.cat(
                            (final_outputs[:max_similarity_idx], final_outputs[max_similarity_idx + 1:]))
                        new_exemplar_set[label].append(class_map[label][max_similarity_idx])
                        class_map[label] = np.delete(class_map[label], max_similarity_idx.cpu().numpy(), axis=0)
        return new_exemplar_set

    def alpha_build_memory(self, data_list, label_list, net, number_of_exemplars_per_class, NUM_CLASSES,
                           old_exemplarset=None):
        if len(old_exemplarset) == 0:
            torch.cuda.empty_cache()
            # to find the exemplars
            new_exemplar_set = dict.fromkeys(np.arange(0, NUM_CLASSES))
            for label in new_exemplar_set:
                new_exemplar_set[label] = []
            # build a dict that form like {classes : data1, data2, ...}
            class_map = fillClassMap(data_list, label_list, NUM_CLASSES)
            # get and save net outputs for each class
            net.eval()
            for label in class_map:
                print('\r'f'building exemplars for class {label}', end='', flush=True)
                if len(class_map[label]) < number_of_exemplars_per_class:
                    for data in class_map[label]:
                        new_exemplar_set[label].append(data)
                else:
                    class_outputs = []
                    mean = 0
                    # calculate class mean
                    with torch.no_grad():
                        class_data_dataset = class_dataset(class_map[label], label)
                        class_data_dataloader = DataLoader(dataset=class_data_dataset, batch_size=64, shuffle=False,
                                                           drop_last=False, pin_memory=True)
                        for class_datas in class_data_dataloader:
                            class_datas_outputs = net(class_datas.to(device), features=True)
                            for output in class_datas_outputs:
                                class_outputs.append(output.unsqueeze(0))
                                mean += output
                            final_outputs = torch.cat(class_outputs, dim=0)
                        # get mean per class
                        mean = mean / len(class_map[label])
                        w_t = mean
                        for i in range(number_of_exemplars_per_class):
                            similarity_list = F.cosine_similarity(w_t, final_outputs)
                            max_similarity_idx = torch.argmax(similarity_list)
                            w_t = w_t + mean - class_outputs[max_similarity_idx]
                            class_outputs.pop(max_similarity_idx)
                            final_outputs = torch.cat(
                                (final_outputs[:max_similarity_idx], final_outputs[max_similarity_idx + 1:]))
                            new_exemplar_set[label].append(class_map[label][max_similarity_idx])
                            class_map[label] = np.delete(class_map[label], max_similarity_idx.cpu().numpy(), axis=0)
            return new_exemplar_set
        else:
            '''find every old_exemplarset_means[label]'''
            old_exemplarset_means = compute_exemplars_mean(old_exemplarset, net, NUM_CLASSES, device)
            torch.cuda.empty_cache()

            # to find the exemplars
            new_exemplar_set = dict.fromkeys(np.arange(0, NUM_CLASSES))
            for label in new_exemplar_set:
                new_exemplar_set[label] = []
            # build a dict that form like {classes : data1, data2, ...}
            class_map = fillClassMap(data_list, label_list, NUM_CLASSES)
            # get and save net outputs for each class
            net.eval()
            for label in class_map:
                print('\r'f'building exemplars for class {label}', end='', flush=True)
                # 1st calculate new_data mean per class
                class_outputs = []
                new_data_mean = 0
                # calculate class mean
                with torch.no_grad():
                    class_data_dataset = class_dataset(class_map[label], label)
                    class_data_dataloader = DataLoader(dataset=class_data_dataset, batch_size=128, shuffle=False,
                                                       drop_last=False, pin_memory=True)
                    for class_datas in class_data_dataloader:
                        class_datas_outputs = net(class_datas.to(device), features=True)
                        for output in class_datas_outputs:
                            class_outputs.append(output.unsqueeze(0))
                            new_data_mean += output
                        final_outputs = torch.cat(class_outputs, dim=0)
                    # get mean per class
                    new_data_mean = new_data_mean / len(class_map[label])
                # 评估概念漂移严重程度 ---> 记忆程度
                new_time_gap_alpha = F.cosine_similarity(new_data_mean, old_exemplarset_means[label])
                cur_TASK_number_of_exemplars_per_class = \
                np.rint(np.exp(1 / new_time_gap_alpha.cpu().numpy())).astype(int)[0]

                if len(class_map[label]) < cur_TASK_number_of_exemplars_per_class:
                    for data in class_map[label]:
                        new_exemplar_set[label].append(data)
                else:
                    w_t = new_data_mean
                    for i in range(cur_TASK_number_of_exemplars_per_class):
                        similarity_list = F.cosine_similarity(w_t, final_outputs)
                        max_similarity_idx = torch.argmax(similarity_list)
                        w_t = w_t + new_data_mean - class_outputs[max_similarity_idx]
                        class_outputs.pop(max_similarity_idx)
                        final_outputs = torch.cat(
                            (final_outputs[:max_similarity_idx], final_outputs[max_similarity_idx + 1:]))
                        new_exemplar_set[label].append(class_map[label][max_similarity_idx])
                        class_map[label] = np.delete(class_map[label], max_similarity_idx.cpu().numpy(), axis=0)
            return new_exemplar_set
