import torch.utils.data
import numpy as np
import pickle
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

randomseed = 42
import domain_incremental_utils

def load_AWF_closeworld_100w():
    '''针对 AWF 100 dataset '''

    # closeworld
    closeworld_200w = np.load('/root/autodl-tmp/DATASET/AWF_dataset/CloseWorld/tor_100w_2500tr.npz',allow_pickle=True)
    closeworld_200w_data = closeworld_200w['data']
    closeworld_200w_real_labels = closeworld_200w['labels']  # real_labels = website name

    # 构建name list 数据集
    closeworld_200w_real_labels_namelist = []
    for name in closeworld_200w_real_labels:
        if name not in closeworld_200w_real_labels_namelist:
            closeworld_200w_real_labels_namelist.append(name)
    label_encoder = LabelEncoder()
    number2name = label_encoder.fit(closeworld_200w_real_labels_namelist)
    closeworld_200w_data_label = number2name.transform(closeworld_200w_real_labels)
    
    return closeworld_200w_data, closeworld_200w_data_label

class WFDataset(Dataset):
    # webiste fingerprint dataset
    def __init__(self, X, Y):
        # filefolder_path = '/root/autodl-tmp/data/CloseWorld/'
        X = X.astype(np.float32)
        Y = torch.tensor(Y, dtype=torch.long)
        self.x_data = X.reshape(X.shape[0], 1, X.shape[1])
        self.y_label = Y
        self.len = X.shape[0]

    def __getitem__(self, index):
        return self.x_data[index], self.y_label[index]

    def __len__(self):
        return self.len

    def set_classes_range(self, low_range, high_range):
        self.low_range = low_range
        self.high_range = high_range

        if low_range != high_range:
            '''
            np.where(condition) 当where内只有一个参数时，那个参数表示条件，当条件成立时，where返回的是每个符合condition条件元素的坐标,返回的是以元组的形式
            '''
            idxes = np.where(np.logical_and(self.y_label >= low_range, self.y_label < high_range))[0]
            self.new_traindata = self.x_data[idxes]
            self.new_trainlabel = self.y_label[idxes]
            self._mapping = {
                fake_idx: real_idx
                for fake_idx, real_idx in enumerate(idxes)
            }
            return idxes
        else:
            idxes = np.where(self.y_label == low_range)[0]
            specific_class_data = self.x_data[idxes]
            specific_class_label = self.y_label[idxes]
            self._mapping = {
                fake_idx: real_idx
                for fake_idx, real_idx in enumerate(idxes)
            }
            return idxes
        # ------------- 筛选新类别的数据 -----------------#
        # 根据标签筛选数据
        # desired_labels = set()  # 0 到 10 的标签
        # indices = [i for i, per_class in enumerate(labels) if per_class in desired_labels]
        # self.new_traindata =
        # self.new_trainlabel =

    def set_examplars(self, idxes):
        '''这段代码用于扩展 _mapping 字典，通过将新的虚拟索引与现有的实际索引关联起来，使得类实例能够跟踪更多的映射关系，通常用于在数据处理中管理索引的映射关系'''
        self._mapping.update({
            fake_idx: real_idx
            for fake_idx, real_idx in zip(range(len(self._mapping), len(idxes)), idxes)
        })

    def get_true_index(self, fake_idx):
        return self._mapping[fake_idx]

def load_DF_closeworld_WalkieTalkie():
    '''针对 DF closeworld walkietalkie dataset '''
    filefolder_path = '/root/autodl-tmp/DATASET/DF_dataset/CloseWorld/WalkieTalkie/'

    with open(filefolder_path + 'X_train_WalkieTalkie.pkl', 'rb') as file:
        X_train = np.array(pickle.load(file, encoding='iso-8859-1'))

    with open(filefolder_path + 'X_test_WalkieTalkie.pkl', 'rb') as file:
        X_test = np.array(pickle.load(file, encoding='iso-8859-1'))

    with open(filefolder_path + 'X_valid_WalkieTalkie.pkl', 'rb') as file:
        X_valid = np.array(pickle.load(file, encoding='iso-8859-1'))

    with open(filefolder_path + 'y_train_WalkieTalkie.pkl', 'rb') as file:
        Y_train = np.array(pickle.load(file, encoding='iso-8859-1'))

    with open(filefolder_path + 'y_test_WalkieTalkie.pkl', 'rb') as file:
        Y_test = np.array(pickle.load(file, encoding='iso-8859-1'))

    with open(filefolder_path + 'y_valid_WalkieTalkie.pkl', 'rb') as file:
        Y_valid = np.array(pickle.load(file, encoding='iso-8859-1'))
    
    '''
    X_train ndarray (80000,5000) -- Y_train ndarry(80000,)
    X_test ndarray (5000,5000) -- Y_test ndarray (5000,)
    X_valid ndarray (5000,5000) -- Y_valid ndarray (5000,)
    '''

    # return X_train, X_valid, X_test, Y_train, Y_valid, Y_test
    return X_train, X_test,Y_train, Y_test


def load_DF_closeworld_WTFPAD():
    '''针对 DF closeworld WTFPAD dataset '''
    filefolder_path = '/root/autodl-tmp/DATASET/DF_dataset/CloseWorld/WTFPAD/'

    with open(filefolder_path + 'X_train_WTFPAD.pkl', 'rb') as file:
        X_train = np.array(pickle.load(file, encoding='iso-8859-1'))

    with open(filefolder_path + 'X_test_WTFPAD.pkl', 'rb') as file:
        X_test = np.array(pickle.load(file, encoding='iso-8859-1'))

    with open(filefolder_path + 'X_valid_WTFPAD.pkl', 'rb') as file:
        X_valid = np.array(pickle.load(file, encoding='iso-8859-1'))

    with open(filefolder_path + 'y_train_WTFPAD.pkl', 'rb') as file:
        Y_train = np.array(pickle.load(file, encoding='iso-8859-1'))

    with open(filefolder_path + 'y_test_WTFPAD.pkl', 'rb') as file:
        Y_test = np.array(pickle.load(file, encoding='iso-8859-1'))

    with open(filefolder_path + 'y_valid_WTFPAD.pkl', 'rb') as file:
        Y_valid = np.array(pickle.load(file, encoding='iso-8859-1'))


    '''
    X_train ndarray (76000,5000) -- Y_train ndarry(76000,)
    X_test ndarray (9500,5000) -- Y_test ndarray (9500,)
    X_valid ndarray (9500,5000) -- Y_valid ndarray (9500,)
    '''
    
    # return X_train, X_valid, X_test, Y_train, Y_valid, Y_test
    return X_train, X_test,Y_train, Y_test


# ---------------------------------AWF dataset --------------------------------------------- #
def get_labelencoder():
    closewordl_data, closewordl_real_label = load_AWF_closeworld_200w()
    concept_data, concept_real_label = load_AWF_conceptdrift_200w('3d')

    # 构建name list 数据集
    namelist = []
    for name in closewordl_real_label:
        if name not in namelist:
            namelist.append(name)
    for name in concept_real_label:
        if name not in namelist:
            namelist.append(name)

    # 构建name list 数据集
    label_encoder = LabelEncoder()
    number2name = label_encoder.fit(namelist)
    return number2name

def load_AWF_closeworld_200w():
    '''针对 AWF 200 dataset '''

    # closeworld
    closeworld_200w = np.load('/root/autodl-tmp/DATASET/AWF_dataset/CloseWorld/tor_200w_2500tr.npz',allow_pickle=True)
    closeworld_200w_data = closeworld_200w['data']
    closeworld_200w_real_labels = closeworld_200w['labels']  # real_labels = website name

    """
    there are some website that concept datasets don't have 
    we need to remove them
    """
    closeworld_removelist = ['extratorrent.cc','feedly.com','mercadolivre.com.br','ok.ru','onclkds.com','yts.ag','sabah.com.tr','thewhizmarketing.com','txxx.com','daikynguyenvn.com','irctc.co.in','mailchimp.com','porn555.com','savefrom.net','themeforest.net','trello.com','wittyfeed.com','xnxx.com','youporn.com','discordapp.com','nicovideo.jp']
    for name in closeworld_removelist:
        idx = np.where(closeworld_200w_real_labels == name)
        closeworld_200w_data = np.delete(closeworld_200w_data, idx, axis=0)
        closeworld_200w_real_labels = np.delete(closeworld_200w_real_labels, idx, axis=0)

    # # 构建name list 数据集
    # closeworld_200w_real_labels_namelist = []
    # for name in closeworld_200w_real_labels:
    #     if name not in closeworld_200w_real_labels_namelist:
    #         closeworld_200w_real_labels_namelist.append(name)
    # label_encoder = LabelEncoder()
    # number2name = label_encoder.fit(closeworld_200w_real_labels_namelist)
    # closeworld_200w_data_label = number2name.transform(closeworld_200w_real_labels)
    #
    # return closeworld_200w_data, closeworld_200w_data_label
    return closeworld_200w_data, closeworld_200w_real_labels

# conceptdrift
def load_AWF_conceptdrift_200w(delay='3day'):
    concept_path = '/root/autodl-tmp/DATASET/AWF_dataset/ConceptDrift/tor_time_test'
    if delay == '3d':
        final_path = concept_path + '3d_200w_100tr.npz'
    elif delay == '10d':
        final_path = concept_path + '10d_200w_100tr.npz'
    elif delay == '2w':
        final_path = concept_path + '2w_200w_100tr.npz'
    elif delay == '4w':
        final_path = concept_path + '4w_200w_100tr.npz'
    elif delay == '6w':
        final_path = concept_path + '6w_200w_100tr.npz'
    else:
        print(f'error can not load{delay}')
        return 0
    data_npz = np.load(final_path,allow_pickle=True)
    data = data_npz['data']
    real_labels = data_npz['labels']

    '''
    there are some website that concept datasets don't have 
    we need to remove them
    '''
    concept_removelist = ['extratorrent.cc','feedly.com','mercadolivre.com.br','ok.ru','onclkds.com','yts.ag','sabah.com.tr','thewhizmarketing.com','txxx.com','daikynguyenvn.com','irctc.co.in','mailchimp.com','porn555.com','savefrom.net','themeforest.net','trello.com','wittyfeed.com','xnxx.com','youporn.com','discordapp.com','nicovideo.jp']
    for name in concept_removelist:
        idx = np.where(real_labels == name)
        real_labels = np.delete(real_labels, idx, axis=0)
        data = np.delete(data, idx, axis=0)

    # # 构建name list 数据集
    # namelist = []
    # for name in real_labels:
    #     if name not in namelist:
    #         namelist.append(name)
    # label_encoder = LabelEncoder()
    # number2name = label_encoder.fit(namelist)
    # data_labels = number2name.transform(real_labels)

    return data, real_labels




def get_AWF_closeword_loader(name2number,batchsize: int, shuffle: bool):
    data, real_label = load_AWF_closeworld_200w()
    label = name2number.transform(real_label)
    # 打乱并切分数据集
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=randomseed,
                                                        shuffle=True)

    train_dataset = WFDataset(X_train, y_train)
    test_dataset = WFDataset(X_test, y_test)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batchsize, pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batchsize, pin_memory=True)

    return train_loader, test_loader, X_train, y_train, X_test, y_test
def get_AWF_Concept_Drift_loader(name2number,batchsize: int, delay:str,shuffle: bool):
    data, real_label = load_AWF_conceptdrift_200w(delay)
    label = name2number.transform(real_label)
    # 打乱并切分数据集
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=randomseed,
                                                        shuffle=True)

    train_dataset = WFDataset(X_train, y_train)
    test_dataset = WFDataset(X_test, y_test)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batchsize, pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batchsize, pin_memory=True)

    return train_loader, test_loader, X_train, y_train, X_test, y_test
# ----------------------------------------------------------------------------------------- #

class current_dataset(torch.utils.data.Dataset):
    """新增数据的data——ground truth"""

    def __init__(self, X, Y):
        # filefolder_path = '/root/autodl-tmp/data/CloseWorld/'
        X = X.astype(np.float32)
        Y = torch.tensor(Y, dtype=torch.long)
        self.x_data = X.reshape(X.shape[0], 1, X.shape[1])
        self.y_label = Y
        self.len = X.shape[0]

    def __getitem__(self, index):
        return self.x_data[index], self.y_label[index]

    def __len__(self):
        return self.len
    

class class_dataset(torch.utils.data.Dataset):
    """单一类别的data——ground truth"""

    def __init__(self, X, Y):
        # filefolder_path = '/root/autodl-tmp/data/CloseWorld/'
        X = X.astype(np.float32)
        # Y = torch.tensor(Y, dtype=torch.long)
        self.x_data = X.reshape(X.shape[0], 1, X.shape[1])
        self.y_label = Y
        self.len = X.shape[0]

    def __getitem__(self, index):
        return self.x_data[index]

    def __len__(self):
        return self.len


def load_ALL_dataloader():

    # load data, label for all datasets
    AWF_closeword_200w_data,AWF_closeword_200w_website_name = load_AWF_closeworld_200w()

    AWF_concept_3d_data, AWF_concept_3d_name = load_AWF_conceptdrift_200w('3d')
    AWF_concept_10d_data, AWF_concept_10d_name = load_AWF_conceptdrift_200w('10d')
    AWF_concept_2w_data, AWF_concept_2w_name = load_AWF_conceptdrift_200w('2w')
    AWF_concept_4w_data, AWF_concept_4w_name = load_AWF_conceptdrift_200w('4w')
    AWF_concept_6w_data, AWF_concept_6w_name = load_AWF_conceptdrift_200w('6w')

    namelist = []
    # 构建name list 数据集
    for subset in [AWF_closeword_200w_website_name,AWF_concept_3d_name,AWF_concept_10d_name,AWF_concept_2w_name,AWF_concept_4w_name,AWF_concept_6w_name]:
        for website_name in subset:
            if website_name not in namelist:
                namelist.append(website_name)
    # build website name to label encoder
    label_encoder = LabelEncoder()
    name2label = label_encoder.fit(namelist)

    AWF_closeword_200w_label = name2label.transform(AWF_closeword_200w_website_name)
    AWF_concept_3d_label = name2label.transform(AWF_concept_3d_name)
    AWF_concept_10d_label = name2label.transform(AWF_concept_10d_name)
    AWF_concept_2w_label = name2label.transform(AWF_concept_2w_name)
    AWF_concept_4w_label = name2label.transform(AWF_concept_4w_name)
    AWF_concept_6w_label = name2label.transform(AWF_concept_6w_name)

    return AWF_closeword_200w_data, AWF_closeword_200w_label, AWF_concept_3d_data, AWF_concept_3d_label, AWF_concept_10d_data, AWF_concept_10d_label, AWF_concept_2w_data, AWF_concept_2w_label,AWF_concept_4w_data, AWF_concept_4w_label,AWF_concept_6w_data, AWF_concept_6w_label




