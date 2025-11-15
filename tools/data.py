#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import pickle
import torch
import numpy as np
from PIL import Image
from torch.utils.data import sampler, DataLoader
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

# In[ ]:


class SVHNDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __getitem__(self, idx):
        label = torch.tensor(self.labels[idx]).to(torch.long)
        image = self.images[idx]      
        image = self.transform(Image.fromarray(image))
        return image, label

    def __len__(self):
        return len(self.labels)


class CIFARDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __getitem__(self, idx):
        label = torch.tensor(self.labels[idx]).to(torch.long)
        image = self.images[idx]      
        image = self.transform(Image.fromarray((255 * image).astype('uint8')))
        
        return image, label

    def __len__(self):
        return len(self.labels)
# In[ ]:


def get_dataloaders(dataset, batch_size):

    # choose the training and test datasets
    #train_data, test_data = get_dataset(dataset)

    # obtain training indices that will be used for validation
    if dataset == 'mnist' or dataset == 'fashion_mnist':
        train_data, test_data = get_dataset(dataset)
        file = open(dataset, 'rb')
        permutations = pickle.load(file)
        indices = permutations['1']
        train_idx, valid_idx, calib_idx= indices[:50000], indices[50000:55000], indices[55000:]
        
    elif dataset == 'svhn':
        # convert data to torch.FloatTensor
        transform_train = transforms.Compose([
            transforms.RandomCrop(32),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
    
        data = np.load('svhn.npy', allow_pickle=True).tolist()
        X, y = data['X'], data['y']
        
        trial = data['t1']
        
        train_indices, test_indices = trial['train_indices'], trial['test_indices']
        
        train_images, train_labels = X[train_indices[:64000]], y[train_indices[:64000]]
        cal_images, cal_labels = X[train_indices[64000:71000]], y[train_indices[64000:71000]]
        val_images, val_labels = X[train_indices[71000:]], y[train_indices[71000:]]
        test_images, test_labels = X[test_indices], y[test_indices]

        train_dataset = SVHNDataset(train_images, train_labels, transform=transform_train)
        cal_dataset = SVHNDataset(cal_images, cal_labels, transform=transform_test)
        val_dataset = SVHNDataset(val_images, val_labels, transform=transform_test)
        test_dataset = SVHNDataset(test_images, test_labels, transform=transform_test)
        
    elif dataset == 'cifar100':
        # convert data to torch.FloatTensor           
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
    
        data = np.load('cifar100.npy', allow_pickle=True).tolist()
        X, y = data['X'].transpose(0,2,3,1), data['y']
        
        trial = data['t2']
        
        train_indices, test_indices = trial['train_indices'], trial['test_indices']
        
        train_images, train_labels = X[train_indices[:40000]], y[train_indices[:40000]]
        cal_images, cal_labels = X[train_indices[40000:45000]], y[train_indices[40000:45000]]
        val_images, val_labels = X[train_indices[45000:]], y[train_indices[45000:]]
        test_images, test_labels = X[test_indices], y[test_indices]

        train_dataset = CIFARDataset(train_images, train_labels, transform=transform_train)
        cal_dataset = CIFARDataset(cal_images, cal_labels, transform=transform_test)
        val_dataset = CIFARDataset(val_images, val_labels, transform=transform_test)
        test_dataset = CIFARDataset(test_images, test_labels, transform=transform_test)        
   
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    calib_loader = DataLoader(cal_dataset, batch_size=len(cal_labels), shuffle=False)
    valid_loader = DataLoader(val_dataset, batch_size=len(val_labels), shuffle=False)

    test_loader = DataLoader(test_dataset, batch_size=len(test_labels), shuffle=False)

    
    return train_loader, calib_loader, test_loader, valid_loader

