#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.gate_layer import GateLayer


# In[2]:


class VGG11(nn.Module):
    def __init__(self, num_classes=1000):
        super(VGG11, self).__init__()
        self.num_classes = num_classes
        # convolutional layers 
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.gate1 = GateLayer(64, 64, [1, -1, 1, 1])
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.gate2 = GateLayer(128, 128, [1, -1, 1, 1])
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.gate3 = GateLayer(256, 256, [1, -1, 1, 1])
        
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(256)
        self.gate4 = GateLayer(256, 256, [1, -1, 1, 1])
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(512)
        self.gate5 = GateLayer(512, 512, [1, -1, 1, 1])
        
        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(512)
        self.gate6 = GateLayer(512, 512, [1, -1, 1, 1])
        self.maxpool6 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv7 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False)
        self.bn7 = nn.BatchNorm2d(512)
        self.gate7 = GateLayer(512, 512, [1, -1, 1, 1])
        
        self.conv8 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False)
        self.bn8 = nn.BatchNorm2d(512)
        self.gate8 = GateLayer(512, 512, [1, -1, 1, 1])
        self.maxpool8 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.avgpool = nn.AvgPool2d(kernel_size=1, stride=1)
        
        self.fc = nn.Linear(512, self.num_classes)
        

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.gate1(out)
        out = F.relu(out)
        out = self.maxpool1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.gate2(out)
        out = F.relu(out)
        out = self.maxpool2(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.gate3(out)
        out = F.relu(out)
        
        out = self.conv4(out)
        out = self.bn4(out)
        out = self.gate4(out)
        out = F.relu(out)
        out = self.maxpool4(out)
        
        out = self.conv5(out)
        out = self.bn5(out)
        out = self.gate5(out)
        out = F.relu(out)
        
        out = self.conv6(out)
        out = self.bn6(out)
        out = self.gate6(out)
        out = F.relu(out)
        out = self.maxpool6(out)
        
        out = self.conv7(out)
        out = self.bn7(out)
        out = self.gate7(out)
        out = F.relu(out)
        
        out = self.conv8(out)
        out = self.bn8(out)
        out = self.gate8(out)
        out = F.relu(out)
        out = self.maxpool8(out)
        
        out = self.avgpool(out)
        
        # flatten to prepare for the fully connected layers
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        
        return out


class VGG16(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG16, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.gate1 = GateLayer(64, 64, [1, -1, 1, 1])
        
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.gate2 = GateLayer(64, 64, [1, -1, 1, 1])
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)        
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.gate3 = GateLayer(128, 128, [1, -1, 1, 1])
       
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(128)
        self.gate4 = GateLayer(128, 128, [1, -1, 1, 1])
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(256)
        self.gate5 = GateLayer(256, 256, [1, -1, 1, 1])        

        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(256)
        self.gate6 = GateLayer(256, 256, [1, -1, 1, 1])
  
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.bn7 = nn.BatchNorm2d(256)
        self.gate7 = GateLayer(256, 256, [1, -1, 1, 1])
        self.maxpool7 = nn.MaxPool2d(kernel_size=2, stride=2)        

        self.conv8 = nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False)
        self.bn8 = nn.BatchNorm2d(512)
        self.gate8 = GateLayer(512, 512, [1, -1, 1, 1])
        
        self.conv9 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False)
        self.bn9 = nn.BatchNorm2d(512)
        self.gate9 = GateLayer(512, 512, [1, -1, 1, 1])

        self.conv10 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False)
        self.bn10 = nn.BatchNorm2d(512)
        self.gate10 = GateLayer(512, 512, [1, -1, 1, 1])
        self.maxpool10 = nn.MaxPool2d(kernel_size=2, stride=2)        
        
        self.conv11 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False)
        self.bn11 = nn.BatchNorm2d(512)
        self.gate11 = GateLayer(512, 512, [1, -1, 1, 1])

        self.conv12 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False)
        self.bn12 = nn.BatchNorm2d(512)
        self.gate12 = GateLayer(512, 512, [1, -1, 1, 1])
        
        self.conv13 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False)
        self.bn13 = nn.BatchNorm2d(512)
        self.gate13 = GateLayer(512, 512, [1, -1, 1, 1])
        self.maxpool13 = nn.MaxPool2d(kernel_size=2, stride=2)  
        
        #self.fc = nn.Linear(512, num_classes)
        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, num_classes))

        
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.gate1(out)
        out = F.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.gate2(out)
        out = F.relu(out)
        out = self.maxpool2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.gate3(out)
        out = F.relu(out)
        
        out = self.conv4(out)
        out = self.bn4(out)
        out = self.gate4(out)
        out = F.relu(out)
        out = self.maxpool4(out)
        
        out = self.conv5(out)
        out = self.bn5(out)
        out = self.gate5(out)
        out = F.relu(out)
        
        out = self.conv6(out)
        out = self.bn6(out)
        out = self.gate6(out)
        out = F.relu(out)

        out = self.conv7(out)
        out = self.bn7(out)
        out = self.gate7(out)
        out = F.relu(out)
        out = self.maxpool7(out)
        
        out = self.conv8(out)
        out = self.bn8(out)
        out = self.gate8(out)
        out = F.relu(out)

        out = self.conv9(out)
        out = self.bn9(out)
        out = self.gate9(out)
        out = F.relu(out)
        
        out = self.conv10(out)
        out = self.bn10(out)
        out = self.gate10(out)
        out = F.relu(out)
        out = self.maxpool10(out)        
        
        out = self.conv11(out)
        out = self.bn11(out)
        out = self.gate11(out)
        out = F.relu(out)

        out = self.conv12(out)
        out = self.bn12(out)
        out = self.gate12(out)
        out = F.relu(out)
        
        out = self.conv13(out)
        out = self.bn13(out)
        out = self.gate13(out)
        out = F.relu(out)
        out = self.maxpool13(out)
        
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)

        
        return out
        
class VGG19(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG19, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.gate1 = GateLayer(64, 64, [1, -1, 1, 1])
        
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.gate2 = GateLayer(64, 64, [1, -1, 1, 1])
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)  

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.gate3 = GateLayer(128, 128, [1, -1, 1, 1])
       
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(128)
        self.gate4 = GateLayer(128, 128, [1, -1, 1, 1])
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(256)
        self.gate5 = GateLayer(256, 256, [1, -1, 1, 1])        

        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(256)
        self.gate6 = GateLayer(256, 256, [1, -1, 1, 1])
  
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.bn7 = nn.BatchNorm2d(256)
        self.gate7 = GateLayer(256, 256, [1, -1, 1, 1])
        
        self.conv8 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.bn8 = nn.BatchNorm2d(256)
        self.gate8 = GateLayer(256, 256, [1, -1, 1, 1])        
        self.maxpool8 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv9 = nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False)
        self.bn9 = nn.BatchNorm2d(512)
        self.gate9 = GateLayer(512, 512, [1, -1, 1, 1])
        
        self.conv10 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False)
        self.bn10 = nn.BatchNorm2d(512)
        self.gate10 = GateLayer(512, 512, [1, -1, 1, 1])
        
        self.conv11 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False)
        self.bn11 = nn.BatchNorm2d(512)
        self.gate11 = GateLayer(512, 512, [1, -1, 1, 1])

        self.conv12 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False)
        self.bn12 = nn.BatchNorm2d(512)
        self.gate12 = GateLayer(512, 512, [1, -1, 1, 1])
        self.maxpool12 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv13 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False)
        self.bn13 = nn.BatchNorm2d(512)
        self.gate13 = GateLayer(512, 512, [1, -1, 1, 1])

        self.conv14 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False)
        self.bn14 = nn.BatchNorm2d(512)
        self.gate14 = GateLayer(512, 512, [1, -1, 1, 1])
        
        self.conv15 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False)
        self.bn15 = nn.BatchNorm2d(512)
        self.gate15 = GateLayer(512, 512, [1, -1, 1, 1])        
        
        self.conv16 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False)
        self.bn16 = nn.BatchNorm2d(512)
        self.gate16 = GateLayer(512, 512, [1, -1, 1, 1])
        self.maxpool16 = nn.MaxPool2d(kernel_size=2, stride=2)  
        
        self.fc = nn.Linear(512, num_classes)


    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.gate1(out)
        out = F.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.gate2(out)
        out = F.relu(out)
        out = self.maxpool2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.gate3(out)
        out = F.relu(out)
        
        out = self.conv4(out)
        out = self.bn4(out)
        out = self.gate4(out)
        out = F.relu(out)
        out = self.maxpool4(out)

        out = self.conv5(out)
        out = self.bn5(out)
        out = self.gate5(out)
        out = F.relu(out)
        
        out = self.conv6(out)
        out = self.bn6(out)
        out = self.gate6(out)
        out = F.relu(out)

        out = self.conv7(out)
        out = self.bn7(out)
        out = self.gate7(out)
        out = F.relu(out)
        
        out = self.conv8(out)
        out = self.bn8(out)
        out = self.gate8(out)
        out = F.relu(out)
        out = self.maxpool8(out)

        out = self.conv9(out)
        out = self.bn9(out)
        out = self.gate9(out)
        out = F.relu(out)
        
        out = self.conv10(out)
        out = self.bn10(out)
        out = self.gate10(out)
        out = F.relu(out)        
        
        out = self.conv11(out)
        out = self.bn11(out)
        out = self.gate11(out)
        out = F.relu(out)

        out = self.conv12(out)
        out = self.bn12(out)
        out = self.gate12(out)
        out = F.relu(out)
        out = self.maxpool12(out)
        
        out = self.conv13(out)
        out = self.bn13(out)
        out = self.gate13(out)
        out = F.relu(out)
        
        out = self.conv14(out)
        out = self.bn14(out)
        out = self.gate14(out)
        out = F.relu(out)        
        
        out = self.conv15(out)
        out = self.bn15(out)
        out = self.gate15(out)
        out = F.relu(out)

        out = self.conv16(out)
        out = self.bn16(out)
        out = self.gate16(out)
        out = F.relu(out)
        out = self.maxpool16(out)        
        
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        
        return out
        
# In[ ]:




