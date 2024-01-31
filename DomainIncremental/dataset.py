import torch.utils.data
import numpy as np
import pickle
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# -------------------------------------------------- #
# ----------------- DFdataset ---------------------- #
# -------------------------------------------------- #
'''close world dataset'''
def load_DF_closeworld_WTFPAD():
    pass
def load_DF_closeworld_WalkieTalkie():
    pass
def load_DF_closeworld_NoDef():
    pass

'''open world dataset'''
def load_DF_openworld_Nodef_trian_evaluate():
    # 针对 DF openworld Nodef dataset trian and evaluation
    filefolder_path = '/root/autodl-tmp/DATASET/DF_dataset/OpenWorld/NoDef/'
    with open(filefolder_path + 'X_train_NoDef.pkl', 'rb') as file:
        X_train = np.array(pickle.load(file, encoding='iso-8859-1'))
    with open(filefolder_path + 'X_valid_NoDef.pkl', 'rb') as file:
        X_valid = np.array(pickle.load(file, encoding='iso-8859-1'))
    with open(filefolder_path + 'y_train_NoDef.pkl', 'rb') as file:
        Y_train = np.array(pickle.load(file, encoding='iso-8859-1'))
    with open(filefolder_path + 'y_valid_NoDef.pkl', 'rb') as file:
        Y_valid = np.array(pickle.load(file, encoding='iso-8859-1'))
    '''
    X_train ndarray (96000,5000) -- Y_train ndarry(96000,)
    X_valid ndarray (5000,5000) -- Y_valid ndarray (5000,)
    '''
    return X_train, X_valid, Y_train, Y_valid

def load_DF_openworld_Nodef_test():
    # 针对 DF openworld Nodef dataset test
    filefolder_path = '/root/autodl-tmp/DATASET/DF_dataset/OpenWorld/NoDef/'

    with open(filefolder_path + 'X_test_Mon_NoDef.pkl', 'rb') as file:
        X_test_Mon = np.array(pickle.load(file, encoding='iso-8859-1'))
    with open(filefolder_path + 'X_test_Unmon_NoDef.pkl', 'rb') as file:
        X_test_Unmon = np.array(pickle.load(file, encoding='iso-8859-1'))
    with open(filefolder_path + 'y_test_Mon_NoDef.pkl', 'rb') as file:
        y_test_Mon = np.array(pickle.load(file, encoding='iso-8859-1'))
    with open(filefolder_path + 'y_test_Unmon_NoDef.pkl', 'rb') as file:
        y_test_Unmon = np.array(pickle.load(file, encoding='iso-8859-1'))
    return X_test_Mon, y_test_Mon, X_test_Unmon, y_test_Unmon

def load_DF_openworld_WalkieTalkie_trian_evaluate():
    # 针对 DF openworld Nodef dataset trian and evaluation
    filefolder_path = '/root/autodl-tmp/DATASET/DF_dataset/OpenWorld/WalkieTalkie/'
    with open(filefolder_path + 'X_train_WalkieTalkie.pkl', 'rb') as file:
        X_train = np.array(pickle.load(file, encoding='iso-8859-1'))
    with open(filefolder_path + 'X_valid_WalkieTalkie.pkl', 'rb') as file:
        X_valid = np.array(pickle.load(file, encoding='iso-8859-1'))
    with open(filefolder_path + 'y_train_WalkieTalkie.pkl', 'rb') as file:
        Y_train = np.array(pickle.load(file, encoding='iso-8859-1'))
    with open(filefolder_path + 'y_valid_WalkieTalkie.pkl', 'rb') as file:
        Y_valid = np.array(pickle.load(file, encoding='iso-8859-1'))
    '''
    X_train ndarray (96000,5000) -- Y_train ndarry(96000,)
    X_valid ndarray (5000,5000) -- Y_valid ndarray (5000,)
    '''
    return X_train, X_valid, Y_train, Y_valid
def load_DF_openworld_WalkieTalkie_test():
    # 针对 DF openworld Nodef dataset test
    filefolder_path = '/root/autodl-tmp/DATASET/DF_dataset/OpenWorld/WalkieTalkie/'

    with open(filefolder_path + 'X_test_Mon_WalkieTalkie.pkl', 'rb') as file:
        X_test_Mon = np.array(pickle.load(file, encoding='iso-8859-1'))
    with open(filefolder_path + 'X_test_Unmon_WalkieTalkie.pkl', 'rb') as file:
        X_test_Unmon = np.array(pickle.load(file, encoding='iso-8859-1'))
    with open(filefolder_path + 'y_test_Mon_WalkieTalkie.pkl', 'rb') as file:
        y_test_Mon = np.array(pickle.load(file, encoding='iso-8859-1'))
    with open(filefolder_path + 'y_test_Unmon_WalkieTalkie.pkl', 'rb') as file:
        y_test_Unmon = np.array(pickle.load(file, encoding='iso-8859-1'))
    return X_test_Mon, y_test_Mon, X_test_Unmon, y_test_Unmon

def load_DF_openworld_WTFPAD_trian_evaluate():
    # 针对 DF openworld Nodef dataset trian and evaluation
    filefolder_path = '/root/autodl-tmp/DATASET/DF_dataset/OpenWorld/WTFPAD/'
    with open(filefolder_path + 'X_train_WTFPAD.pkl', 'rb') as file:
        X_train = np.array(pickle.load(file, encoding='iso-8859-1'))
    with open(filefolder_path + 'X_valid_WTFPAD.pkl', 'rb') as file:
        X_valid = np.array(pickle.load(file, encoding='iso-8859-1'))
    with open(filefolder_path + 'y_train_WTFPAD.pkl', 'rb') as file:
        Y_train = np.array(pickle.load(file, encoding='iso-8859-1'))
    with open(filefolder_path + 'y_valid_WTFPAD.pkl', 'rb') as file:
        Y_valid = np.array(pickle.load(file, encoding='iso-8859-1'))
    '''
    X_train ndarray (96000,5000) -- Y_train ndarry(96000,)
    X_valid ndarray (5000,5000) -- Y_valid ndarray (5000,)
    '''
    return X_train, X_valid, Y_train, Y_valid
def load_DF_openworld_WTFPAD_test():
    # 针对 DF openworld Nodef dataset test
    filefolder_path = '/root/autodl-tmp/DATASET/DF_dataset/OpenWorld/WTFPAD/'

    with open(filefolder_path + 'X_test_Mon_WTFPAD.pkl', 'rb') as file:
        X_test_Mon = np.array(pickle.load(file, encoding='iso-8859-1'))
    with open(filefolder_path + 'X_test_Unmon_WTFPAD.pkl', 'rb') as file:
        X_test_Unmon = np.array(pickle.load(file, encoding='iso-8859-1'))
    with open(filefolder_path + 'y_test_Mon_WTFPAD.pkl', 'rb') as file:
        y_test_Mon = np.array(pickle.load(file, encoding='iso-8859-1'))
    with open(filefolder_path + 'y_test_Unmon_WTFPAD.pkl', 'rb') as file:
        y_test_Unmon = np.array(pickle.load(file, encoding='iso-8859-1'))
    return X_test_Mon, y_test_Mon, X_test_Unmon, y_test_Unmon


# -------------------------------------------------- #
# ----------------- AWFdataset --------------------- #
# -------------------------------------------------- #
'''close world'''
def load_AWF_closeworld_200w_realnamelabel():
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
    return closeworld_200w_data, closeworld_200w_real_labels

def load_AWF_closeworld_100w_realnamelabel():
    pass
def load_AWF_closeworld_500w_realnamelabel():
    pass
def load_AWF_closeworld_900w_realnamelabel():
    pass
def load_AWF_closeworld_200w():
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

def load_AWF_closeworld_100w():
    pass
def load_AWF_closeworld_500w():
    pass
def load_AWF_closeworld_900w():
    pass

'''concept drift'''
def load_AWF_conceptdrift_200w_realnamelabel(delay='3day'):
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
    return data, real_labels

'''open world'''
def load_AWF_openworld_200w():
    openworld_path = '/root/autodl-tmp/DATASET/AWF_dataset/OpenWorld/tor_open_200w_2000tr.npz'
    data_npz = np.load(openworld_path)
    data = data_npz['data']
    real_labels = data_npz['labels']
    return data, real_labels
def load_AWF_openworld_400000w():
    openworld_path = '/root/autodl-tmp/DATASET/AWF_dataset/OpenWorld/tor_open_400000w.npz'
    data_npz = np.load(openworld_path)
    data = data_npz['data']
    real_labels = data_npz['labels']
    return data, real_labels
