import os
import random
import json
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
import pydicom

import medmnist
from medmnist import INFO, Evaluator


class MedMNISTDataLoader(DataLoader):
    def __init__(self, data_flag, batch_size=128, shuffle=True, num_workers=4):
        info = INFO[data_flag] # {'python_class': 'DermaMNIST', 'description': 'The DermaMNIST is based on the HAM10000, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. The dataset consists of 10,015 dermatoscopic images categorized as 7 different diseases, formulized as a multi-class classification task. We split the images into training, validation and test set with a ratio of 7:1:2. The source images of 3×600×450 are resized into 3×28×28.', 'url': 'https://zenodo.org/record/6496656/files/dermamnist.npz?download=1', 'MD5': '0744692d530f8e62ec473284d019b0c7', 'task': 'multi-class', 'label': {'0': 'actinic keratoses and intraepithelial carcinoma', '1': 'basal cell carcinoma', '2': 'benign keratosis-like lesions', '3': 'dermatofibroma', '4': 'melanoma', '5': 'melanocytic nevi', '6': 'vascular lesions'}, 'n_channels': 3, 'n_samples': {'train': 7007, 'val': 1003, 'test': 2005}, 'license': 'CC BY-NC 4.0'}
        self.task = info['task']
        n_channels = info['n_channels']
        self.n_classes = len(info['label'])

        DataClass = getattr(medmnist, info['python_class'])

        # preprocessing
        data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])
        ])

        # load the data
        train_dataset = DataClass(split='train', transform=data_transform, download=True)
        val_dataset = DataClass(split='val', transform=data_transform, download=True)
        test_dataset = DataClass(split='test', transform=data_transform, download=True)

        # encapsulate data into dataloader form
        self.train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        self.val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        self.train_loader_at_eval = DataLoader(dataset=train_dataset, batch_size=2*batch_size, shuffle=False, num_workers=num_workers)
        self.test_loader = DataLoader(dataset=test_dataset, batch_size=2*batch_size, shuffle=False, num_workers=num_workers)
    
    def get_train_dataloader(self):
        return self.train_loader

    def get_val_dataloader(self):
        return self.val_loader
    
    def get_train_dataloader_at_eval(self):
        return self.train_loader_at_eval
    
    def get_test_dataloader(self):
        return self.test_loader

  
"""
train_dataset:
Using downloaded and verified file: /home/lh9998/scratch/lh9998/medmnist/dermamnist.npz
Dataset DermaMNIST (dermamnist)
    Number of datapoints: 7007
    Root location: /home/lh9998/scratch/lh9998/medmnist
    Split: train
    Task: multi-class
    Number of channels: 3
    Meaning of labels: {'0': 'actinic keratoses and intraepithelial carcinoma', '1': 'basal cell carcinoma', '2': 'benign keratosis-like lesions', '3': 'dermatofibroma', '4': 'melanoma', '5': 'melanocytic nevi', '6': 'vascular lesions'}
    Number of samples: {'train': 7007, 'val': 1003, 'test': 2005}
    Description: The DermaMNIST is based on the HAM10000, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. The dataset consists of 10,015 dermatoscopic images categorized as 7 different diseases, formulized as a multi-class classification task. We split the images into training, validation and test set with a ratio of 7:1:2. The source images of 3×600×450 are resized into 3×28×28.
    License: CC BY-NC 4.0
===================
test_dataset:
Dataset DermaMNIST (dermamnist)
    Number of datapoints: 2005
    Root location: /home/lh9998/scratch/lh9998/medmnist
    Split: test
    Task: multi-class
    Number of channels: 3
    Meaning of labels: {'0': 'actinic keratoses and intraepithelial carcinoma', '1': 'basal cell carcinoma', '2': 'benign keratosis-like lesions', '3': 'dermatofibroma', '4': 'melanoma', '5': 'melanocytic nevi', '6': 'vascular lesions'}
    Number of samples: {'train': 7007, 'val': 1003, 'test': 2005}
    Description: The DermaMNIST is based on the HAM10000, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. The dataset consists of 10,015 dermatoscopic images categorized as 7 different diseases, formulized as a multi-class classification task. We split the images into training, validation and test set with a ratio of 7:1:2. The source images of 3×600×450 are resized into 3×28×28.
    License: CC BY-NC 4.0
"""

class AugmentedMedMNISTDataset(Dataset):
    def __init__(self, image_folder, train, transform=None):
        self.transform = transform
        if train: # training dataset
            self.image_folder = image_folder + 'train/'
        else: # testing dataset
            self.image_folder = image_folder + 'test/'
        
        metadata_file = self.image_folder + 'metadata.csv'
        metadata_df = pd.read_csv(metadata_file)
        metadata_df = metadata_df.reset_index(drop=True)

        all_classes, self.total_classes_sizes = np.unique(metadata_df["labels"], return_counts=True)
        self.total_num_classes = all_classes.shape[0]
        
        self.targets = metadata_df['labels']
        self.img_names = metadata_df['image_name']
            
    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        image = Image.open(self.image_folder+self.img_names[idx], mode='r') # 28, 28
        if self.transform:
            image = self.transform(image)
        label = self.targets[idx]
        return image, label
    
class AugmentedMedMNISTDataLoader(DataLoader):
    def __init__(self, image_folder, batch_size=128, shuffle=True, num_workers=4):
        data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])
        ])

        # load the data
        train_dataset = AugmentedMedMNISTDataset(image_folder, True, data_transform)
        test_dataset = AugmentedMedMNISTDataset(image_folder, False, data_transform)

        # encapsulate data into dataloader form
        self.train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        self.train_loader_at_eval = DataLoader(dataset=train_dataset, batch_size=2*batch_size, shuffle=False, num_workers=num_workers)
        self.test_loader = DataLoader(dataset=test_dataset, batch_size=2*batch_size, shuffle=False, num_workers=num_workers)
        # self.n_classes = train_dataset.total_num_classes # 5?
        self.n_classes = 7
    
    def get_train_dataloader(self):
        return self.train_loader
    
    def get_train_dataloader_at_eval(self):
        return self.train_loader_at_eval
    
    def get_test_dataloader(self):
        return self.test_loader
    

class ISIC2020Dataset(Dataset):
    def __init__(self, train, transform=None):
        self.isic_folder = '/home/lh9998/scratch/lh9998/isic2020/'
        self.image_folder = 'train/'
        metadata_file_train = 'ISIC_2020_Training_GroundTruth_v2.csv'
        # metadata_file_test = 'ISIC_2020_Test_Metadata.csv' # no gt label
        
        self.metadata_df, self.total_num_classes, self.total_classes_sizes = self.process_metadata(self.isic_folder, metadata_file_train)
        self.img_names = self.metadata_df['image_name']
        self.transform = transform
        self.targets = self.metadata_df['labels']
        
        train_inds, test_inds = self.stratify_train_test(self.targets, self.total_num_classes)
        if train: # training dataset
            self.targets = self.targets[train_inds]
            self.img_names = self.img_names[train_inds]
            self.targets = self.targets.reset_index(drop=True)
            self.img_names = self.img_names.reset_index(drop=True)
        else: # testing dataset
            self.targets = self.targets[test_inds]
            self.img_names = self.img_names[test_inds]
            self.targets = self.targets.reset_index(drop=True)
            self.img_names = self.img_names.reset_index(drop=True)
        
    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        img_path = self.isic_folder+self.image_folder+self.img_names[idx]+'.dcm'
        dcm_img = pydicom.dcmread(img_path, force=True) # dcm_img.pixel_array.size = product(dcm_img.pixel_array.shape)
        # print(dcm_img.Rows, dcm_img.Columns, dcm_img.pixel_array.shape, dcm_img.pixel_array.size) # 480 640; 1053 1872; 6000 4000
        image = Image.fromarray(dcm_img.pixel_array)
        # image = Image.open(self.isic_folder+self.image_folder+self.img_names[idx]+'.dcm', mode='r')
        # print("image.shape", image.size)
        if self.transform:
            image = self.transform(image)
        # print("image.shape after", image.shape) # torch.Size([3, 128, 128])
        label = self.targets[idx]
        return image, label

    def process_metadata(self, isic_folder, metadata_file):
        target_col = 'benign_malignant'
        metadata_df = pd.read_csv(isic_folder+metadata_file) # diagnosis: 6 types. biggest class is 'unknown'
        metadata_df = metadata_df[metadata_df[target_col].notnull()] # training: (33126, 9)
        metadata_df = metadata_df.sort_values(target_col)
        metadata_df = metadata_df.reset_index(drop=True)

        all_classes, class_counts = np.unique(metadata_df[target_col], return_counts=True)
        count_sort_ind = np.argsort(-class_counts) # sort from most to least
        class_counts_sorted = -np.sort(-class_counts)
        all_classes = all_classes[count_sort_ind]
        num_classes = all_classes.shape[0]
        # print(all_classes, class_counts_sorted) # ['benign' 'malignant'] [32542   584]
        metadata_df['labels'] = metadata_df[target_col].replace(all_classes,list(range(len(all_classes)))) # convert str labels to int
        return metadata_df, num_classes, class_counts_sorted

    def stratify_train_test(self, targets, num_classes):
        random.seed(123)
        class_inds = {}
        for idx in range(num_classes):
            class_inds[idx] = []
        for i in range(len(targets)): 
            class_inds[targets[i]].append(i)

        train_inds = set()
        test_inds = set()
        for c in class_inds.keys():
            train_inds_temp = random.sample(class_inds[c], int(len(class_inds[c])*0.7))
            test_inds.update([item for item in class_inds[c] if item not in train_inds_temp])
            train_inds.update([item for item in class_inds[c] if item in train_inds_temp])
        test_inds = list(test_inds)
        train_inds = list(train_inds)
        return train_inds, test_inds

class ISIC2020DataLoader(DataLoader):
    """
    Load ISIC 2020
    """
    def __init__(self, batch_size=32, shuffle=True, num_workers=4):
        # normalize = transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
        #     std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
        train_trsfm = transforms.Compose([
            transforms.Resize(128),
            transforms.RandomCrop(128),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation((0,180)),
            transforms.ToTensor(),
            # normalize,
            transforms.Normalize(mean=[.5], std=[.5])
        ])
        test_trsfm = transforms.Compose([
            transforms.Resize(128),
            transforms.CenterCrop(128),
            transforms.ToTensor(),
            # normalize,
            transforms.Normalize(mean=[.5], std=[.5])
        ])
        
        self.train_dataset = ISIC2020Dataset(train=True, transform=train_trsfm)
        self.test_dataset = ISIC2020Dataset(train=False, transform=test_trsfm)
        
        self.train_loader = DataLoader(dataset=self.train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        self.train_loader_at_eval = DataLoader(dataset=self.train_dataset, batch_size=2*batch_size, shuffle=False, num_workers=num_workers)
        self.test_loader = DataLoader(dataset=self.test_dataset, batch_size=2*batch_size, shuffle=False, num_workers=num_workers)
        self.n_classes = self.train_dataset.total_num_classes
    
    def get_train_dataloader(self):
        return self.train_loader

    def get_train_dataloader_at_eval(self):
        return self.train_loader_at_eval
    
    def get_test_dataloader(self):
        return self.test_loader


if __name__ == '__main__':
    invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0.],
                                                     std = [ 1/0.5 ]),
                                transforms.Normalize(mean = [ -0.5 ],
                                                     std = [ 1. ]),
                               ])
    
    loader = MedMNISTDataLoader('dermamnist')
    train_dataloader = loader.get_train_dataloader()
    val_dataloader = loader.get_val_dataloader()
    train_dataloader_at_eval = loader.get_train_dataloader_at_eval()
    test_dataloader = loader.get_test_dataloader()
    
    # # isicdataset = ISIC2020Dataset(True, None)
    # loader = ISIC2020DataLoader(batch_size=32, shuffle=True, num_workers=4)
    # train_dataloader = loader.get_train_dataloader()
    # train_dataloader_at_eval = loader.get_train_dataloader_at_eval()
    # test_dataloader = loader.get_test_dataloader()
    
    # loader = AugmentedMedMNISTDataLoader("/home/lh9998/exp3/slot-attention-pytorch/models_tmp/generator_2023_10_03_03_55_52/")
    # train_dataloader = loader.get_train_dataloader()
    # train_dataloader_at_eval = loader.get_train_dataloader_at_eval()
    # test_dataloader = loader.get_test_dataloader()
    
    for images, labels in val_dataloader:
        images = invTrans(images)
        print(images.shape, images)
        print(torch.min(images), torch.max(images))
        exit(0)
    
