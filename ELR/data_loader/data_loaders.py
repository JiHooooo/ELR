import sys
from torchvision import datasets, transforms
from base import BaseDataLoader
from data_loader.cifar10 import get_cifar10
from data_loader.cifar100 import get_cifar100
from parse_config import ConfigParser
from PIL import Image
import pandas as pd
import random
from .SelfDataset import SelfDataset
from .augmentation import get_augmentation

class CIFAR10DataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_batches=0,  training=True, num_workers=4,  pin_memory=True):
        config = ConfigParser.get_instance()
        cfg_trainer = config['trainer']
        
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        self.data_dir = data_dir

        noise_file='%sCIFAR10_%.1f_Asym_%s.json'%(config['data_loader']['args']['data_dir'],cfg_trainer['percent'],cfg_trainer['asym'])
        
        self.train_dataset, self.val_dataset = get_cifar10(config['data_loader']['args']['data_dir'], cfg_trainer, train=training,
                                                           transform_train=transform_train, transform_val=transform_val, noise_file = noise_file)

        super().__init__(self.train_dataset, batch_size, shuffle, validation_split, num_workers, pin_memory,
                         val_dataset = self.val_dataset)
    def run_loader(self, batch_size, shuffle, validation_split, num_workers, pin_memory):
        super().__init__(self.train_dataset, batch_size, shuffle, validation_split, num_workers, pin_memory,
                         val_dataset = self.val_dataset)



class CIFAR100DataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_batches=0, training=True,num_workers=4, pin_memory=True):
        config = ConfigParser.get_instance()
        cfg_trainer = config['trainer']
        
        transform_train = transforms.Compose([
                #transforms.ColorJitter(brightness= 0.4, contrast= 0.4, saturation= 0.4, hue= 0.1),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ])
        transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        self.data_dir = data_dir
        config = ConfigParser.get_instance()
        cfg_trainer = config['trainer']

        noise_file='%sCIFAR100_%.1f_Asym_%s.json'%(config['data_loader']['args']['data_dir'],cfg_trainer['percent'],cfg_trainer['asym'])

        self.train_dataset, self.val_dataset = get_cifar100(config['data_loader']['args']['data_dir'], cfg_trainer, train=training,
                                                           transform_train=transform_train, transform_val=transform_val, noise_file = noise_file)

        super().__init__(self.train_dataset, batch_size, shuffle, validation_split, num_workers, pin_memory,
                         val_dataset = self.val_dataset)
    def run_loader(self, batch_size, shuffle, validation_split, num_workers, pin_memory):
        super().__init__(self.train_dataset, batch_size, shuffle, validation_split, num_workers, pin_memory,
                         val_dataset = self.val_dataset)

class SelfDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, 
                shuffle=True, validation_split=0.0, num_batches=0,  
                training=True, num_workers=4,  pin_memory=True):
        config = ConfigParser.get_instance()
        cfg_trainer = config['trainer']

        transforms_train = get_augmentation(input_size=(config['data_loader']['args']['image_size'],config['data_loader']['args']['image_size']),
                                            train_flag=True,
                                            normalize_flag = config['data_loader']['args']['normalize_flag'])

        transforms_val = get_augmentation(input_size=(config['data_loader']['args']['image_size'],config['data_loader']['args']['image_size']),
                                            train_flag=False,
                                            normalize_flag = config['data_loader']['args']['normalize_flag'])

        self.train_dataset = SelfDataset(config['data_loader']['args']['data_dir']+'/train',
                                        config['data_loader']['args']['label_name'], train=training, 
                                        change_ratio=cfg_trainer['percent'], transforms=transforms_train)
        self.val_dataset = SelfDataset(config['data_loader']['args']['data_dir']+'/val', 
                                        config['data_loader']['args']['label_name'], train=False, 
                                        change_ratio=cfg_trainer['percent'], transforms=transforms_val)
        
        
        super().__init__(self.train_dataset, batch_size, shuffle, validation_split, num_workers, pin_memory,
                         val_dataset = self.val_dataset)
    def run_loader(self, batch_size, shuffle, validation_split, num_workers, pin_memory):
        super().__init__(self.train_dataset, batch_size, shuffle, validation_split, num_workers, pin_memory,
                         val_dataset = self.val_dataset)

def Image_Info_from_df(df_path):
    print('start loading information')
    df = pd.read_csv(df_path,encoding="cp932")
    image_info = []
    for index in range(len(df)):
        #input image name
        image_info_one = [df.iloc[index]['image_path'],]
        for label in _label_name:
            image_info_one.append(int(df.iloc[index][label]))                                 
        image_info.append(image_info_one)
    print('finish loading information')
    return image_info

def save_train_val(info, save_path):
    info_columns = ['image_path',]
    info_columns.extend(_label_name)
    info_dict = {name:[] for name in info_columns}
    for one_info in info:
        for column_index, column in enumerate(info_columns):
            info_dict[column].append(one_info[column_index])
    pd.DataFrame(info_dict).to_csv(save_path)