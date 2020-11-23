# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 09:51:58 2020

DataLoader of EIT multi-level data.

@author: XIANG

"""
from torch.utils.data import Dataset, DataLoader
import numpy as np
from glob import glob
from os.path import dirname, join as pjoin
from torch.utils.data.sampler import SubsetRandomSampler


class EITDataset(Dataset):
    """Prepare EIT Dataset"""
    
    def __init__(self, mode, data_dir, transform=None):
        assert mode in ['train','test'] # training, test and validate dataset;

        self.transform = transform
        if mode == 'train': # training and validation dataset; 1,2,4 circles
            self.input_ = []
            fle = pjoin(data_dir, "measV1.npy")
            self.input_.append(np.load(fle))
            fle = pjoin(data_dir, "measV2.npy")
            self.input_.append(np.load(fle))
            fle = pjoin(data_dir, "measV3.npy")
            self.input_.append(np.load(fle))    
            fle = pjoin(data_dir, "measV4.npy")
            self.input_.append(np.load(fle))
            self.input_ = np.concatenate(self.input_)

            self.target_ = []
            fle = pjoin(data_dir, "img1.npy")
            self.target_.append(np.load(fle))
            fle = pjoin(data_dir, "img2.npy")
            self.target_.append(np.load(fle))
            fle = pjoin(data_dir, "img3.npy")
            self.target_.append(np.load(fle))
            fle = pjoin(data_dir, "img4.npy")
            self.target_.append(np.load(fle))
            self.target_ = np.concatenate(self.target_)

            # for fle in glob(pjoin(data_dir, "measV*[1,2,4].npy")):
            #     self.input_.append(np.load(fle))
            # self.input_ = np.concatenate(self.input_)
            
            # self.target_ = []
            # for fle in glob(pjoin(data_dir, "img*[1,2,4].npy")):
            #     self.target_.append(np.load(fle))
            # self.target_ = np.concatenate(self.target_)
            
        else: # testing validation: 3 circles
            self.input_ = np.load(pjoin(data_dir, 'measV4.npy'))
            self.target_ = np.load(pjoin(data_dir, 'img4.npy'))
                  
            
    def __len__(self):
        return np.size(self.input_,0)
    
    def __getitem__(self, idx):
        input_m, target_img = self.input_[idx], self.target_[idx]
        # transform the input tensor into required formats
        if self.transform:
            input_m = self.transform(input_m)
        
        # target image range [0, +1]
        return (input_m,  - target_img)

def get_loader(mode='train', data_dir=None, transform=None, batch_size=128, num_workers=4):
    dataset_ = EITDataset(mode=mode, data_dir=data_dir, transform=transform)
    print('Total', mode, ' data size: ', len(dataset_))
    data_loader = DataLoader(dataset=dataset_, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return data_loader


def DataSplit(root_dir, batch_size=128, validation_split=0.2, transform=None):
    shuffle_dataset=True
    random_seed=42
    dataset = EITDataset(mode='train', data_dir=root_dir, transform=transform)
    test_loader = get_loader(mode='test', data_dir=root_dir, batch_size=batch_size, transform=transform)
    
    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    split = int(np.floor(validation_split * dataset_size))
    train_indices, val_indices = indices[split:], indices[:split]
        
    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)
    
    return train_loader, val_loader, test_loader




