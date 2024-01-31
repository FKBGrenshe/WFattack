from sklearn.preprocessing import LabelEncoder
import numpy as np



def load_AWF_closeworld_100w():
    print("loading AWF_closeworld_100w dataset...")
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
    
    return np.array(closeworld_200w_data), np.array(closeworld_200w_data_label)




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
    label_list = []
    for label in closeworld_200w_real_labels:
        if label not in label_list:
            label_list.append(label)
    label_encoder = LabelEncoder()
    number2name = label_encoder.fit(label_list)
    closeworld_200w_data_label = number2name.transform(closeworld_200w_real_labels)
    return np.array(closeworld_200w_data), np.array(closeworld_200w_data_label)
   

def load_AWF_concept_drift(delay='3day'):
    concept_path = '/root/autodl-tmp/DATASET/AWF_dataset/ConceptDrift/tor_time_test'
    if delay == '3day':
        final_path = concept_path + '3d_200w_100tr.npz'
    elif delay == '10day':
        final_path = concept_path + '10d_200w_100tr.npz'
    elif delay == '2week':
        final_path = concept_path + '2w_200w_100tr.npz'
    elif delay == '4week':
        final_path = concept_path + '4w_200w_100tr.npz'
    elif delay == '6week':
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
    namelist = []
    for name in real_labels:
        if name not in namelist:
            namelist.append(name)
    label_encoder = LabelEncoder()
    number2name = label_encoder.fit(namelist)
    data_labels = number2name.transform(real_labels)

    return np.array(data), np.array(data_labels)

def load_AWF_closeworld_200w_realname():
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

    return data, real_labels

def load_ALL_dataloader():

    # load data, label for all datasets
    AWF_closeword_200w_data,AWF_closeword_200w_website_name = load_AWF_closeworld_200w_realname()

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
