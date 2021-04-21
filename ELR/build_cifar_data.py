import torchvision
import PIL.Image as Image
import numpy as np
import tqdm
import os

def save_image(np_array, index_list, save_folder, label_name,ext='bmp'):
    for one_img, index in tqdm.tqdm(zip(np_array, index_list)):
        Image.fromarray(one_img).save('%s/%s_%04d.%s'%(save_folder, label_name, index, ext))

save_folder = 'data'
sub_folder = ['train', 'val']

dataset = torchvision.datasets.CIFAR10(save_folder, train=True, download=True)
dataset_val = torchvision.datasets.CIFAR10(save_folder, train=False, download=True)

label_name_dist = {j:i for i,j in dataset.class_to_idx.items()}
for dataset_one, sub_folder_one in zip([dataset, dataset_val], sub_folder):
    for label_index, label_name in label_name_dist.items():
        save_folder_one = '%s/%s/%s'%(save_folder, sub_folder_one, label_name)
        os.makedirs(save_folder_one)
        one_type_data = dataset_one.data[np.where(np.array(dataset_one.targets) == label_index)]
        save_image(one_type_data, [i for i in range(len(one_type_data))], save_folder_one, label_name)
