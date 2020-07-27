import os
from glob import glob
import numpy as np
from torch.utils.data import Dataset, DataLoader

import os
from glob import glob
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

class ct_dataset(Dataset):
    def __init__(self, mode, load_mode, ds_factor, saved_path, test_patient,  transform=None):
        assert mode in ['train', 'test'], "mode is 'train' or 'test'"
        assert load_mode in [0,1], "load_mode is 0 or 1"

        input_path = sorted(glob(os.path.join(saved_path, '*_input.npy')))
        target_path = sorted(glob(os.path.join(saved_path, '*_target.npy')))
        self.load_mode = load_mode
        self.transform = transform

        if mode == 'train':
            input_ = [f for f in input_path if test_patient not in f]
            input_ = [f for f in input_ if '_ds'+str(ds_factor)+'_' in f]
            target_ = [f for f in target_path if test_patient not in f]
            target_ = [f for f in target_ if '_ds'+str(ds_factor)+'_' in f]

            if load_mode == 0: # batch data load
                self.input_ = input_
                self.target_ = target_
            else: # all data load
                self.input_ = [np.load(f) for f in input_]
                self.target_ = [np.load(f) for f in target_]
        else: # mode =='test'
            # load test set with different downsampling factor
            input_ = [f for f in input_path if test_patient in f]
            input_ = [f for f in input_ if '_ds'+str(ds_factor)+'_' in f]
            target_ = [f for f in target_path if test_patient in f]
            target_ = [f for f in target_ if '_ds'+str(ds_factor)+'_' in f]
            
            if load_mode == 0:
                self.input_ = input_
                self.target_ = target_
            else:
                self.input_ = [np.load(f) for f in input_]
                self.target_ = [np.load(f) for f in target_]

    def __len__(self):
        return len(self.target_)

    def __getitem__(self, idx):
        input_img, target_img = self.input_[idx], self.target_[idx]
        if self.load_mode == 0:
            input_img, target_img = np.load(input_img), np.load(target_img)

        if self.transform:
            input_img = self.transform(input_img)
            target_img = self.transform(target_img)


        return (input_img, target_img)


def get_loader(mode='train', load_mode=0, ds_factor=6, saved_path=None, test_patient='L311',transform=None,batch_size=32):
    dataset_ = ct_dataset(mode, load_mode, ds_factor, saved_path, test_patient, transform)
    data_loader = DataLoader(dataset=dataset_, batch_size=batch_size, shuffle=True, num_workers=6)
    return data_loader


def DataSplit(test_patient='L311', batch_size=4, validation_split=0.2, ds_factor=6, saved_path=None,  transform=None):
    shuffle_dataset=True
    random_seed=42
    dataset_ = ct_dataset(mode='train', load_mode=0, ds_factor=ds_factor, saved_path=saved_path,
     test_patient=test_patient, transform=transform)

    test_loader = get_loader(mode='test', load_mode=0, ds_factor=ds_factor,
        saved_path=saved_path, test_patient=test_patient,transform=transform,batch_size=batch_size)
    
    # Creating data indices for training and validation splits:
    dataset_size = len(dataset_)
    indices = list(range(dataset_size))
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    split = int(np.floor(validation_split * dataset_size))
    train_indices, val_indices = indices[split:], indices[:split]
        
    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    
    train_loader = DataLoader(dataset_, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(dataset_, batch_size=batch_size, sampler=valid_sampler)
    
    return train_loader, val_loader, test_loader