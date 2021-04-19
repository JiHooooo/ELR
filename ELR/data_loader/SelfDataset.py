import os
import glob
from PIL import Image
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import random

_extension = ['jpg','png','bmp']

class SelfDataset_val(Dataset):
    def __init__(self, folder_name, transforms = None):
        self.folder_name = folder_name
        self.transforms = transforms
        self.image_pathes = self.load_image_of_one_folder(folder_name)

    @staticmethod
    def load_image_of_one_folder(folder_path):
        image_pathes = []
        for root, dirs, files in os.walk(folder_path):
            for file_name in files:
                if os.path.splitext(file_name)[1][1:] in _extension:
                    image_pathes.append('%s/%s'%(root, file_name))
        return image_pathes
    
    def __len__(self):
        return len(self.image_pathes)
    
    def __getitem__(self, idx):
        image_path = self.image_pathes[idx]
        image = Image.open(image_path)
        image = np.array(image, dtype=np.uint8)
        if self.transforms:
            transformed = self.transforms(image=image)
            image = transformed['image']
        else:
            image = (torch.tensor(image)/255.0)
            if len(image.shape) == 3:
                image = image.transpose(2,0,1)
            elif len(image.shape) == 2:
                image = image[None]
            else:
                raise TypeError('The dim of input image should be 2 or 3, but get %d'%(len(image.shape)))
        return image

class SelfDataset_multi(Dataset):
    def __init__(self, folder_name, image_info, label_num, transforms = None, 
                train_flag=True, change_ratio=0):
        self.folder_name = folder_name
        self.image_info = image_info
        self.image_info_gt = image_info.copy()
        self.transforms = transforms
        self.label_num = label_num
        self.change_ratio = change_ratio
        if train_flag and change_ratio > 0:
            self.symmetric_noise()

    def symmetric_noise(self):
        change_num = 0
        indices = np.random.permutation(len(self.image_info))
        for i, index in enumerate(indices):
            if i < self.change_ratio*len(self.image_info):
                exist_label = []
                for index_label in range(self.label_num):
                    if self.image_info[index][index_label+1] > 0:
                        exist_label.append(index_label)
                if len(exist_label)>1:
                    random.shuffle(exist_label)
                    for modify_label in exist_label[1:]:
                        self.image_info[index][modify_label+1] = 0
                    change_num += 1
        print('%d(%d) images have been modified'%(change_num, len(self.image_info)))


    def __len__(self):
        return len(self.image_info)

    def __getitem__(self, idx):
        image_name = self.image_info[idx][0]
        image_path = self.folder_name + '/' +image_name
        image = Image.open(image_path)
        image = np.array(image, dtype=np.uint8)
        
        labels = np.zeros(self.label_num, dtype=np.float32)
        for index_label in range(self.label_num):
            if self.image_info[idx][index_label+1] > 0:
                labels[index_label] = 1
        labels = torch.from_numpy(labels)
        if self.transforms:
            transformed = self.transforms(image=image)
            image = transformed['image']

        return image, labels

## dataset loading method
class SelfDataset(Dataset):
    def __init__(self, folder_path, label_name, train=True, change_ratio=0, transforms = None):

        self.transforms = transforms
        #read all image path
        image_path_list = []
        image_label_list = []
        sub_folder_list = os.listdir(folder_path)
        sub_folder_list.sort()
        self.label_name = label_name
        print('label name list : ' + str(self.label_name))
        for label_folder in sub_folder_list:
            if label_folder in self.label_name:
                image_path_list_one_folder = []
                index_folder = self.label_name.index(label_folder)
                for extension in _extension:
                    image_path_list_one_folder.extend(glob.glob('%s/%s/*.%s'%(folder_path, label_folder, extension)))
                image_label_list_one_folder = [index_folder for _ in range(len(image_path_list_one_folder))]
                image_path_list.extend(image_path_list_one_folder)
                image_label_list.extend(image_label_list_one_folder)

        self.image_path_list = image_path_list
        self.image_label_list = image_label_list
        self.image_label_list_gt = image_label_list.copy()
        self.change_ratio = change_ratio
        if len(self.image_path_list) == 0:
            raise ValueError("Don't find suitable image file")
        if train and change_ratio > 0:
            self.symmetric_noise()

    def __len__(self):
        return len(self.image_path_list)

    def symmetric_noise(self):
        
        indices = np.random.permutation(len(self.image_path_list))
        for i, idx in enumerate(indices):
            if i < self.change_ratio * len(self.image_path_list):
                self.image_label_list[idx] = np.random.randint(len(self.label_name), dtype=np.int32)
    
    def asymmetric_noise(self):
        for i in range(len(self.label_name)):
            indices = np.where(self.image_label_list_gt == i)[0]
            np.random.shuffle(indices)
            for j, idx in enumerate(indices):
                if j < self.change_ratio * len(indices):
                    # truck -> automobile
                    if i == 9:
                        self.image_label_list[idx] = 1
                    # bird -> airplane
                    elif i == 2:
                        self.image_label_list[idx] = 0
                    # cat -> dog
                    elif i == 3:
                        self.image_label_list[idx] = 5
                    # dog -> cat
                    elif i == 5:
                        self.image_label_list[idx] = 3
                    # deer -> horse
                    elif i == 4:
                        self.image_label_list[idx] = 7

    def __getitem__(self, idx):
        #load label
        label = self.image_label_list[idx]
        label = torch.tensor(label).type(torch.int64)
        label_gt = torch.tensor(self.image_label_list_gt[idx]).type(torch.int64)
        #load image
        image_path = self.image_path_list[idx]
        image = Image.open(image_path).convert('RGB')
        image = np.array(image, dtype=np.uint8)
        #image augmentation
        if self.transforms:
            transformed = self.transforms(image=image)
            image = transformed['image']
        return image, label, idx, label_gt

class SelfDataset_fold(SelfDataset):
    def __init__(self, df_path, label_name, train_flag = True,transforms = None, fold_num = 0, train_total=False, change_ratio=0):
        #load transfrom method
        self.transforms = transforms
        #load df information
        df_info = pd.read_csv(df_path)
        print('the label name : ' + str(label_name))
        self.label_name = [str(i) for i in label_name]
        if train_flag:
            if train_total:
                selected_df_info = df_info
            else:
                selected_df_info = df_info[df_info['fold'] != fold_num]
        else:
            selected_df_info = df_info[df_info['fold'] == fold_num]

        self.image_path_list = list(selected_df_info['image_path'])
        self.image_label_list = [self.label_name.index(i) for i in list(selected_df_info['label'])]
        self.image_label_list_gt = self.image_label_list.copy()
        self.change_ratio = change_ratio
        if len(self.image_path_list) == 0:
            raise ValueError("Don't find suitable image file")
        if train_flag and change_ratio > 0:
            self.symmetric_noise()

def Image_Info_from_df(df_path):
    print('start loading information')
    df = pd.read_csv(df_path)
    image_info = []
    for index in range(len(df)):
        image_info_one_folder = [df.iloc[index]['image_path'], df.iloc[index]['label']]
        image_info.append(image_info_one_folder)
    print('finish loading information')
    return image_info

def save_fold_info(df_info, valid_index_list, save_path=None):
    df_info_new = df_info.copy()
    df_info_new['fold'] = 0
    for fold_index, valid_indexes in enumerate(valid_index_list):
        for valid_index in valid_indexes:
            df_info_new.iloc[valid_index, 'fold'] = fold_index
    if save_path is not None:
        df_info_new.save(save_path)
    return df_info_new

