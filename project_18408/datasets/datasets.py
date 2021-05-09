# -*- coding: utf-8 -*-
"""
Created on Sat May  8 15:05:40 2021

@author: Shahir
"""

import os
import math
from functools import partial

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import torchvision
from torchvision import transforms
import torchvision.transforms.functional as TF

class DatasetType:
    MNIST = "mnist"
    CIFAR10 = "cifar10"
    CIFAR100 = "cifar100"

DATASET_TO_NUM_CLUSTERS = {
    DatasetType.MNIST: 10,
    DatasetType.CIFAR10: 10,
    DatasetType.CIFAR100: 100
}

DATASET_TO_IMG_SIZE = {
    DatasetType.MNIST: 28,
    DatasetType.CIFAR10: 32,
    DatasetType.CIFAR100: 32
}

class ImageDataset(Dataset):
    def __init__(self, x_data, y_data=None, metadata=None):
        self.labeled = y_data is not None
        self.x_data = x_data
        self.y_data = y_data
        self.metadata = metadata

    def __getitem__(self, index):
        if self.labeled:
            return (self.x_data[index], self.y_data[index])
        
        return self.x_data[index]

    def __len__(self):
        return len(self.x_data)

class TransformDataset(Dataset):
    def __init__(self, dataset, transform_func, labeled=True):
        self.dataset = dataset
        self.transform_func = transform_func
        self.labeled = labeled

    def __getitem__(self, index):
        if self.labeled:
            original_img, target = self.dataset[index]
            img = self.transform_func(original_img)
            return (img, target)
        else:
            original_img = self.dataset[index]
            img = self.transform_func(original_img)
            return img
    
    def __len__(self):
        return len(self.dataset)

def apply_transforms(datasets,
                     dataset_type,
                     new_input_size,
                     augment=False): 
    if dataset_type == DatasetType.MNIST:
        train_transform = transforms.Compose([
            transforms.Resize(new_input_size),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.125),
            transforms.RandomApply([
                transforms.RandomAffine(degrees=10,
                                        translate=(0.1, 0.1),
                                        scale=(0.9, 1.1),
                                        shear=10)], p=0.9),
            transforms.ToTensor()])
    
        test_transform = transforms.Compose([
            transforms.Resize(new_input_size),
            transforms.ToTensor()])
    
    elif dataset_type == DatasetType.CIFAR10 or dataset_type == DatasetType.CIFAR100:
        train_transform = transforms.Compose([
            transforms.Resize(new_input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.125),
            transforms.RandomApply([
                transforms.RandomAffine(degrees=10,
                                        translate=(0.1, 0.1),
                                        scale=(0.9, 1.1),
                                        shear=10)], p=0.9),
            transforms.ToTensor()])
        
        test_transform = transforms.Compose([
            transforms.Resize(new_input_size),
            transforms.ToTensor()])
    else:
        raise ValueError()
    
    if not augment:
        train_transform = test_transform
    
    train_dataset = TransformDataset(dataset['train'], train_transform)
    test_dataset = TransformDataset(dataset['test'], test_transform)
    
    return {'train': train_dataset,
            'test': test_dataset}

def get_datasets(dataset_type, data_dir):
    if dataset_type == DatasetType.MNIST:
        train_dataset = torchvision.datasets.MNIST(
            data_dir,
            train=True,
            download=True)
        test_dataset = torchvision.datasets.MNIST(
            data_dir,
            train=False,
            download=True)
    elif dataset_type == DatasetType.CIFAR10:
        train_dataset = torchvision.datasets.CIFAR10(
            data_dir,
            train=True,
            download=True)
        test_dataset = torchvision.datasets.CIFAR10(
            data_dir,
            train=False,
            download=True)
    elif dataset_type == DatasetType.CIFAR100:
        train_dataset = torchvision.datasets.CIFAR100(
            data_dir,
            train=True,
            download=True)
        test_dataset = torchvision.datasets.CIFAR100(
            data_dir,
            train=False,
            download=True)
    else:
        raise ValueError()
    
    return {'train': train_dataset,
            'test': test_dataset}

def get_dataloaders(datasets,
                    train_batch_size,
                    test_batch_size,
                    num_workers=4,
                    pin_memory=False):
    train_dataset = datasets['train']
    train_loader = DataLoader(train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory)
    
    test_dataset = datasets['test']
    test_loader = DataLoader(test_dataset,
        batch_size=test_val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory)
    test_shuffle_loader = DataLoader(test_dataset,
        batch_size=test_val_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory)
    
    return {'train': train_loader,
            'test': test_loader,
            'test_shuffled': test_shuffle_loader}
