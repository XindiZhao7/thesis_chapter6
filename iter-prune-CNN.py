#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from tools.test_model import evaluate
from sklearn.preprocessing import StandardScaler
from icp import pValues, calculate_q
from loss import DOICPLoss
from models.resnet import ResNet18, ResNet34
from models.vgg import VGG16, VGG19
from openpyxl import Workbook



if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
print('Using PyTorch version:', torch.__version__, ' Device:', device)


# In[ ]:



def updatemodel(model, mask):
    gate_counter = 0
    for name, param in model.named_parameters():
        if 'gate' in name:
            param.data = mask[gate_counter]
            gate_counter += 1

    return model


def calculate_node_importance(args, model, valid_loader, criterion):
    importances = list()
    
    start=time.time()
    for X_val, y_val in valid_loader:
        bs = args.batch_size
        cat_bs = int(bs/2)
        Xcat, ycat = X_val[:cat_bs].to(device), y_val[:cat_bs].to(device)
        Xtr, ytr = X_val[cat_bs:bs].to(device), y_val[cat_bs:bs].to(device)
        out_cat = model(Xcat)
        cat_ncs = nn.LogSoftmax(dim=1)(out_cat)
        catAlphas = cat_ncs[[torch.arange(0, len(ycat)), ycat]]
        q = calculate_q(catAlphas, args.epsilon)
        out = model(Xtr)
        ncs = nn.LogSoftmax(dim=1)(out)
        # calculate the loss
        loss = criterion(ncs, q)
    
        loss.backward()

    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'gate' in name or 'fc' in name:
                continue
            else:
                if len(param.shape)>1:
                    importance = (param*param.grad).data.pow(2).view(param.shape[0],-1).sum(dim=1)
                    importances.append(importance)
        node_importance = torch.cat(importances, dim=0).cpu().numpy()
    end=time.time()
    
    return node_importance, end-start


def getmask(model, flatten_importances, percentile = 0.2):
    masks = list()
    nunits_by_layer = list()
    
    for name, param in model.named_parameters():
        if 'gate' in name:
            masks.append(param)
        elif 'fc' in name:
            break
        else:
            if len(param.shape)>1:
                nunits = param.shape[0]
                nunits_by_layer.append(nunits)

    flatten_mask = torch.cat(masks, dim=0)
    
    remaining_nodes_num = int(np.sum(flatten_mask.cpu().numpy()))
    remaining_nodes_indices = np.where(flatten_mask.cpu().numpy()==1)[0]
    
    _, indices = torch.sort(flatten_importances[remaining_nodes_indices])
    
    new_mask = flatten_mask.clone()    
    new_mask[remaining_nodes_indices[indices.cpu().numpy()][:int(percentile*len(indices))]]=0
    new_masks=torch.split(new_mask, nunits_by_layer)
    
    percent = torch.sum(new_mask)/len(flatten_mask)

    print('Percentage remaining', percent.cpu().numpy(), end = ' ')
    print('Layer nodes:', remaining_nodes_num, end = ' ')
    print('\n')

    file = open("results/"+str(percent.cpu().numpy())+".txt", "w")
    file.write(str(remaining_nodes_num)+"\n")
    file.close()

    return new_masks,  percent


# In[ ]:


def main():
    parser = argparse.ArgumentParser(description='CP pruning')
    parser.add_argument('--model', default='resnet34', type=str,
                        help='model selection, choices: mlp1, mlp2, vgg16, vgg19, resnet18, resnet34',
                        choices=['abs-cp', 'sign-cp', 'magnitude', 'taylor', 'snr'])
    parser.add_argument('--dataset', default='cifar100', type=str,
                        help='dataset selection, choices: mnist, fashion_mnist, covtype, svhn, tmnist, cifar10, cifar100',
                        choices=['mnist', 'fashion_mnist', 'covtype', 'svhn', 'tmnist', 'cifar10', 'cifar100'])
    parser.add_argument('--feature-dim', type=tuple, default=(32,32,3),
                        help='feature dimension (default: 10)')       
    parser.add_argument('--num-classes', type=int, default=100,
                        help='number of classes (default: 10)')
    parser.add_argument('--pretrained', type=bool, default=True,
                        help='whehter there exists a pretrained model')                           
    parser.add_argument('--batch-size', type=int, default=200,
                        help='input batch size for training (default: 50)')
    parser.add_argument('--loss', type=str, default='CT',
                        help='CE=CrossEntropy, CT=ConfTr')
    parser.add_argument('--epochs', type=int, default=1500,
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--finetune-epochs', type=int, default=300,
                        help='number of epochs to train (default: 200)')    
    parser.add_argument('--patience', type=int, default=60,
                        help='number of epochs for early stopping (default: 50)')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.0005,
                        help='weight decay (default: 0.0)')    
    parser.add_argument('--epsilon', type=int, default=0.01,
                        help='significance level (default: 0.01)')
    parser.add_argument('--save-path', default='results/cifar100-resnet34-ct-t2.pt', type=str,
                        help='save path for model weights')    
    parser.add_argument('--method', default='abs-cp', type=str,
                        help='pruning criterion selection, choices: abs-cp, sign-cp, magnitude, taylor, snr',
                        choices=['abs-cp', 'sign-cp', 'magnitude', 'taylor', 'snr'])
    parser.add_argument('--gamma', type=int, default=5, metavar='M',
                        help='sigmoid function gamma (default: 2)')

    args = parser.parse_args([])
    
    percent = 1.0
    mask = None
    importances = list()
    mean_sizes = list()
    coverages = list()
    accuracys = list()
    timer={'train':[], 'importance_estimation':[], 'finetune':[], 'train_epoch':[], 'finetune_epoch':[]}
    performance={'mean_size':[], 'coverage':[], 'acc':[]}
    
    if args.model=='resnet18':
        model = ResNet18(num_classes=args.num_classes).to(device)
    elif args.model=='resnet34':
        model = ResNet34(num_classes=args.num_classes).to(device)
        hidden_units = [64, 64, 64, 64, 64, 64, 64, 128, 128, 128, 128, 128, 128, 128, 128, 128, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512]

    elif args.model=='vgg16':
        model = VGG16(num_classes=args.num_classes).to(device)
    elif args.model=='vgg19':
        model = VGG19(num_classes=args.num_classes).to(device)
        
    num_hidden_layers = len(hidden_units)
    num_hidden_units = np.sum(hidden_units)
    
    if args.loss == 'CE':
        from tools.vanilla_training import train
        criterion = nn.CrossEntropyLoss()
    elif args.loss == 'CT':
        from tools.conformal_training import train
        criterion = DOICPLoss(gamma=args.gamma)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    
    train_loader, calib_loader, test_loader, valid_loader = get_dataloaders(args.dataset, args.batch_size)
    
    if args.pretrained == True:
        model.load_state_dict(torch.load(args.save_path))
        f = open("results/train_time.txt")
        train_time = f.read()
        f.close()
    else:
        # train model                  
        model, train_time, epoch = train(args, model, train_loader, calib_loader, valid_loader, criterion, optimizer, num_hidden_layers, None, device, finetune=False)
    
    timer['train'].append(train_time)
    timer['train_epoch'].append(args.finetune_epochs+args.patience)
    
    mean_size, coverage, acc = evaluate(model, calib_loader, valid_loader, args.epsilon, device)
    mean_sizes.append(mean_size)
    coverages.append(coverage)
    accuracys.append(acc)

    mean_size, coverage, acc = evaluate(model, calib_loader, test_loader, args.epsilon, device)
    performance['mean_size'].append(mean_size)
    performance['coverage'].append(coverage)
    performance['acc'].append(acc)

    
    print('\n>>> Start Pruning<<<')
    
    while percent > 0.05:
        last_percent = percent
        node_importance, crite_time=calculate_node_importance(args, model, valid_loader, criterion)
        timer['importance_estimation'].append(crite_time)
        
        mask, percent = getmask(model, torch.tensor(node_importance))

        if last_percent == percent:
            break
        model=updatemodel(model, mask)
        print('**************************************************************')
        print('\n>>> Retraining<<<')
        model, train_time, epoch = train(args, model, train_loader, calib_loader, valid_loader, criterion, optimizer, num_hidden_layers, mask, device, finetune=True)
        timer['finetune'].append(train_time)
        timer['finetune_epoch'].append(epoch)
        mean_size, coverage, acc=evaluate(model, calib_loader, test_loader, args.epsilon, device)
        performance['mean_size'].append(mean_size)
        performance['coverage'].append(coverage)
        performance['acc'].append(acc)
        print('**************************************************************')

    
    np.save('performance.npy', performance)
    np.save('time.npy', timer)
    return


# In[2]:


main()



# In[ ]:




