#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import argparse
import pickle
import time
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from tools.data import get_dataloaders
from sklearn.preprocessing import StandardScaler
from icp import pValues, calculate_q
from loss import DOICPLoss
from models.resnet import ResNet18, ResNet34
from models.vgg import VGG16, VGG19
from openpyxl import Workbook


# In[ ]:



if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
print('Using PyTorch version:', torch.__version__, ' Device:', device)


# In[ ]:


def main():
    parser = argparse.ArgumentParser(description='CP pruning')
    parser.add_argument('--model', default='vgg19', type=str,
                        help='model selection, choices: mlp1, mlp2, vgg16, vgg19, resnet18, resnet34',
                        choices=['abs-cp', 'sign-cp', 'magnitude', 'taylor', 'snr'])
    parser.add_argument('--dataset', default='cifar100', type=str,
                        help='dataset selection, choices: mnist, fashion_mnist, covtype, svhn, tmnist, cifar10, cifar100',
                        choices=['mnist', 'fashion_mnist', 'covtype', 'svhn', 'tmnist', 'cifar10', 'cifar100'])
    parser.add_argument('--feature-dim', type=tuple, default=(32,32,3),
                        help='feature dimension (default: 10)')       
    parser.add_argument('--num-classes', type=int, default=100,
                        help='number of classes (default: 10)')    
    parser.add_argument('--batch-size', type=int, default=200,
                        help='input batch size for training (default: 50)')
    parser.add_argument('--loss', type=str, default='CT',
                        help='CE=CrossEntropy, CT=ConfTr')
    parser.add_argument('--epochs', type=int, default=1500,
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--finetune-epochs', type=int, default=1000,
                        help='number of epochs to train (default: 200)')    
    parser.add_argument('--patience', type=int, default=100,
                        help='number of epochs for early stopping (default: 50)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.0,
                        help='weight decay (default: 0.0)')    
    parser.add_argument('--epsilon', type=int, default=0.01,
                        help='significance level (default: 0.01)')
    parser.add_argument('--save-path', default='results/cifar100-vgg19-ct-t2.pt', type=str,
                        help='save path for model weights')    
    parser.add_argument('--method', default='taylor', type=str,
                        help='pruning criterion selection, choices: abs-cp, sign-cp, magnitude, taylor, snr',
                        choices=['abs-cp', 'sign-cp', 'magnitude', 'taylor', 'snr'])
    parser.add_argument('--gamma', type=int, default=5, metavar='M',
                        help='sigmoid function gamma (default: 2)')

    args = parser.parse_args([])
    if args.model=='resnet18':
        model = ResNet18(num_classes=args.num_classes).to(device)
        hidden_units = [64, 64, 64, 64, 64, 128, 128, 128, 128, 128, 256, 256, 256, 256, 256, 512, 512, 512, 512, 512]
    elif args.model=='resnet34':
        model = ResNet34(num_classes=args.num_classes).to(device)
        hidden_units = [64, 64, 64, 64, 64, 64, 64, 128, 128, 128, 128, 128, 128, 128, 128, 128, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512]

    elif args.model=='vgg16':
        model = VGG16(num_classes=args.num_classes).to(device)
    elif args.model=='vgg19':
        model = VGG19(num_classes=args.num_classes).to(device)
        hidden_units = [64, 64, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512, 512]
        
    num_hidden_layers = len(hidden_units)
    
    if args.loss == 'CE':
        from tools.vanilla_training import train
        criterion = nn.CrossEntropyLoss()
    elif args.loss == 'CT':
        from tools.conformal_training import train
        criterion = DOICPLoss(gamma=args.gamma)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    
    train_loader, calib_loader, test_loader, valid_loader = get_dataloaders(args.dataset, args.batch_size)
    
    model, train_time, epoch = train(args, model, train_loader, calib_loader, valid_loader, criterion, optimizer, num_hidden_layers, None, device, finetune=False)
    
    file = open("results/train_time.txt", "w")
    file.write(str(train_time))
    file.close()
    
    file = open("results/train_epoch.txt", "w")
    file.write(str(epoch))
    file.close()
    
    return


# In[ ]:


main()

