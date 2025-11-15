"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

# based on https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.gate_layer import GateLayer




class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.gate1 = GateLayer(64, 64, [1, -1, 1, 1])
        
        # Basic Block 1 (64, 64, 1)
        self.shortcut2 = nn.Sequential()
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.gate2 = GateLayer(64, 64, [1, -1, 1, 1])
        
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.gate3 = GateLayer(64, 64, [1, -1, 1, 1])
        
        # Basic Block 2 (64, 64, 1)
        self.shortcut4 = nn.Sequential()
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(64)
        self.gate4 = GateLayer(64, 64, [1, -1, 1, 1])
        
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(64)
        self.gate5 = GateLayer(64, 64, [1, -1, 1, 1])
        
        # Basic Block 3 (64, 128, 2)
        self.shortgate6 = GateLayer(128, 128, [1, -1, 1, 1])
        self.shortconv6 = nn.Conv2d(64, 128, kernel_size=1, stride=2, bias=False)
        self.shortbn6 = nn.BatchNorm2d(128)
        self.shortcut6 = nn.Sequential(
            self.shortconv6,
            self.shortbn6,
            self.shortgate6
            )
        self.conv6 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(128)
        self.gate6 = GateLayer(128, 128, [1, -1, 1, 1])
        
        self.conv7 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn7 = nn.BatchNorm2d(128)
        self.gate7 = GateLayer(128, 128, [1, -1, 1, 1])
        
        #Basic block 4 (128, 128, 1)
        self.shortcut8 = nn.Sequential()
        self.conv8 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn8 = nn.BatchNorm2d(128)
        self.gate8 = GateLayer(128, 128, [1, -1, 1, 1])
        
        self.conv9 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn9 = nn.BatchNorm2d(128)
        self.gate9 = GateLayer(128, 128, [1, -1, 1, 1])
        
        #Basic block 5 (128, 256, 2)
        self.shortgate10 = GateLayer(256, 256, [1, -1, 1, 1])
        self.shortconv10 = nn.Conv2d(128, 256, kernel_size=1, stride=2, bias=False)
        self.shortbn10 =nn.BatchNorm2d(256) 
        self.shortcut10 = nn.Sequential(
            self.shortconv10,
            self.shortbn10,
            self.shortgate10
            )
        self.conv10 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn10 = nn.BatchNorm2d(256)
        self.gate10 = GateLayer(256, 256, [1, -1, 1, 1])
        
        self.conv11 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn11 = nn.BatchNorm2d(256)
        self.gate11 = GateLayer(256, 256, [1, -1, 1, 1])
        
        #Basic block 6 (256, 256, 1)
        self.shortcut12 = nn.Sequential()
        self.conv12 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn12 = nn.BatchNorm2d(256)
        self.gate12 = GateLayer(256, 256, [1, -1, 1, 1])
        
        self.conv13 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn13 = nn.BatchNorm2d(256)
        self.gate13 = GateLayer(256, 256, [1, -1, 1, 1])
        
        #Basic block 7 (256, 512, 2)
        self.shortgate14 = GateLayer(512, 512, [1, -1, 1, 1])
        self.shortconv14 = nn.Conv2d(256, 512, kernel_size=1, stride=2, bias=False)
        self.shortbn14 = nn.BatchNorm2d(512)
        self.shortcut14 = nn.Sequential(
            self.shortconv14,
            self.shortbn14,
            self.shortgate14
            )
        self.conv14 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn14 = nn.BatchNorm2d(512)
        self.gate14 = GateLayer(512, 512, [1, -1, 1, 1])
        
        self.conv15 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn15 = nn.BatchNorm2d(512)
        self.gate15 = GateLayer(512, 512, [1, -1, 1, 1])
        
        #Basic block 8 (512, 512, 1)
        self.shortcut16 = nn.Sequential()
        self.conv16 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn16 = nn.BatchNorm2d(512)
        self.gate16 = GateLayer(512, 512, [1, -1, 1, 1])
        
        self.conv17 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn17 = nn.BatchNorm2d(512)
        self.gate17 = GateLayer(512, 512, [1, -1, 1, 1])
        
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, gate):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, gate))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.gate1(out)
        out = F.relu(out)
       
        # Basic Block 1
        shortcut = self.shortcut2(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.gate2(out)
        out = F.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.gate3(out)
        out = F.relu(out)
        
        out += shortcut
        
        # Basic Block 2
        shortcut = self.shortcut4(out)
        out = self.conv4(out)
        out = self.bn4(out)
        out = self.gate4(out)
        out = F.relu(out)
        
        out = self.conv5(out)
        out = self.bn5(out)
        out = self.gate5(out)
        out = F.relu(out)
        
        out += shortcut
        
        # Basic Block 3
        shortcut = self.shortcut6(out)
        out = self.conv6(out)
        out = self.bn6(out)
        out = self.gate6(out)
        out = F.relu(out)
        
        out = self.conv7(out)
        out = self.bn7(out)
        out = self.gate7(out)
        out = F.relu(out)

        out += shortcut
        
        # Basic Block 4
        shortcut = self.shortcut8(out)
        out = self.conv8(out)
        out = self.bn8(out)
        out = self.gate8(out)
        out = F.relu(out)
        
        out = self.conv9(out)
        out = self.bn9(out)
        out = self.gate9(out)
        out = F.relu(out)

        out += shortcut
        
        # Basic Block 5
        shortcut = self.shortcut10(out)
        out = self.conv10(out)
        out = self.bn10(out)
        out = self.gate10(out)
        out = F.relu(out)
        
        out = self.conv11(out)
        out = self.bn11(out)
        out = self.gate11(out)
        out = F.relu(out)
        
        out += shortcut
        
        # Basic Block 6
        shortcut = self.shortcut12(out)
        out = self.conv12(out)
        out = self.bn12(out)
        out = self.gate12(out)
        out = F.relu(out)
        
        out = self.conv13(out)
        out = self.bn13(out)
        out = self.gate13(out)
        out = F.relu(out)

        out += shortcut
        
        # Basic Block 7
        shortcut = self.shortcut14(out)
        out = self.conv14(out)
        out = self.bn14(out)
        out = self.gate14(out)
        out = F.relu(out)
        
        out = self.conv15(out)
        out = self.bn15(out)
        out = self.gate15(out)
        out = F.relu(out)

        out += shortcut
        
        # Basic Block 8
        shortcut = self.shortcut16(out)
        out = self.conv16(out)
        out = self.bn16(out)
        out = self.gate16(out)
        out = F.relu(out)
        
        out = self.conv17(out)
        out = self.bn17(out)
        out = self.gate17(out)
        out = F.relu(out)

        out += shortcut
        
        # fc
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out


class ResNet34(nn.Module):
    def __init__(self, num_classes=100):
        super(ResNet34, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.gate1 = GateLayer(64, 64, [1, -1, 1, 1])
        
        # Basic Block 1 (64, 64, 1)
        self.shortcut2 = nn.Sequential()
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.gate2 = GateLayer(64, 64, [1, -1, 1, 1])
        
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.gate3 = GateLayer(64, 64, [1, -1, 1, 1])
        
        # Basic Block 2 (64, 64, 1)
        self.shortcut4 = nn.Sequential()
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(64)
        self.gate4 = GateLayer(64, 64, [1, -1, 1, 1])
        
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(64)
        self.gate5 = GateLayer(64, 64, [1, -1, 1, 1])
        
        # Basic Block 3 (64, 64, 1)
        self.shortcut6 = nn.Sequential()
        self.conv6 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(64)
        self.gate6 = GateLayer(64, 64, [1, -1, 1, 1])
        
        self.conv7 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn7 = nn.BatchNorm2d(64)
        self.gate7 = GateLayer(64, 64, [1, -1, 1, 1])        
        
        # Basic Block 4 (128, 128, 2)
        self.conv8 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn8 = nn.BatchNorm2d(128)
        self.gate8 = GateLayer(128, 128, [1, -1, 1, 1])    

        self.conv9 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn9 = nn.BatchNorm2d(128)
        self.gate9 = GateLayer(128, 128, [1, -1, 1, 1])
        
        self.shortgate8 = GateLayer(128, 128, [1, -1, 1, 1])
        self.shortconv8 = nn.Conv2d(64, 128, kernel_size=1, stride=2, bias=False)
        self.shortbn8 = nn.BatchNorm2d(128)
        self.shortcut8 = nn.Sequential(
            self.shortconv8,
            self.shortbn8,
            self.shortgate8
            ) 

        #Basic block 5 (128, 128, 1)
        self.shortcut10 = nn.Sequential()
        self.conv10 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn10 = nn.BatchNorm2d(128)
        self.gate10 = GateLayer(128, 128, [1, -1, 1, 1])
        
        self.conv11 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn11 = nn.BatchNorm2d(128)
        self.gate11 = GateLayer(128, 128, [1, -1, 1, 1])
        
        
        #Basic block 6 (128, 128, 1)
        self.shortcut12 = nn.Sequential()
        self.conv12 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn12 = nn.BatchNorm2d(128)
        self.gate12 = GateLayer(128, 128, [1, -1, 1, 1])
        
        self.conv13 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn13 = nn.BatchNorm2d(128)
        self.gate13 = GateLayer(128, 128, [1, -1, 1, 1])
        
        #Basic block 7 (128, 128, 1)
        self.shortcut14 = nn.Sequential()
        self.conv14 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn14 = nn.BatchNorm2d(128)
        self.gate14 = GateLayer(128, 128, [1, -1, 1, 1])
        
        self.conv15 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn15 = nn.BatchNorm2d(128)
        self.gate15 = GateLayer(128, 128, [1, -1, 1, 1])

        #Basic block 8 (128, 256, 2)
        self.conv16 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn16 = nn.BatchNorm2d(256)
        self.gate16 = GateLayer(256, 256, [1, -1, 1, 1])       
        
        self.conv17 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn17 = nn.BatchNorm2d(256)
        self.gate17 = GateLayer(256, 256, [1, -1, 1, 1])
        
        self.shortgate16 = GateLayer(256, 256, [1, -1, 1, 1])
        self.shortconv16 = nn.Conv2d(128, 256, kernel_size=1, stride=2, bias=False)
        self.shortbn16 =nn.BatchNorm2d(256) 
        self.shortcut16 = nn.Sequential(
            self.shortconv16,
            self.shortbn16,
            self.shortgate16
            )         
        
        #Basic block 9 (256, 256, 1)
        self.shortcut18 = nn.Sequential()
        self.conv18 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn18 = nn.BatchNorm2d(256)
        self.gate18 = GateLayer(256, 256, [1, -1, 1, 1])
        
        self.conv19 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn19 = nn.BatchNorm2d(256)
        self.gate19 = GateLayer(256, 256, [1, -1, 1, 1])
        
        #Basic block 10 (256, 256, 1)
        self.shortcut20 = nn.Sequential()
        self.conv20 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn20 = nn.BatchNorm2d(256)
        self.gate20 = GateLayer(256, 256, [1, -1, 1, 1])
        
        self.conv21 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn21 = nn.BatchNorm2d(256)
        self.gate21 = GateLayer(256, 256, [1, -1, 1, 1])
        
        #Basic block 11 (256, 256, 1)
        self.shortcut22 = nn.Sequential()
        self.conv22 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn22 = nn.BatchNorm2d(256)
        self.gate22 = GateLayer(256, 256, [1, -1, 1, 1])
        
        self.conv23 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn23 = nn.BatchNorm2d(256)
        self.gate23 = GateLayer(256, 256, [1, -1, 1, 1])
                
        #Basic block 12 (256, 256, 1)
        self.shortcut24 = nn.Sequential()
        self.conv24 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn24 = nn.BatchNorm2d(256)
        self.gate24 = GateLayer(256, 256, [1, -1, 1, 1])
        
        self.conv25 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn25 = nn.BatchNorm2d(256)
        self.gate25 = GateLayer(256, 256, [1, -1, 1, 1])
        
        #Basic block 13 (256, 256, 1)
        self.shortcut26 = nn.Sequential()
        self.conv26 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn26 = nn.BatchNorm2d(256)
        self.gate26 = GateLayer(256, 256, [1, -1, 1, 1])
        
        self.conv27 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn27 = nn.BatchNorm2d(256)
        self.gate27 = GateLayer(256, 256, [1, -1, 1, 1])

        #Basic block 14 (256, 512, 2)
        self.conv28 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn28 = nn.BatchNorm2d(512)
        self.gate28 = GateLayer(512, 512, [1, -1, 1, 1])       
        
        self.conv29 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn29 = nn.BatchNorm2d(512)
        self.gate29 = GateLayer(512, 512, [1, -1, 1, 1])
        
        self.shortgate28 = GateLayer(512, 512, [1, -1, 1, 1])
        self.shortconv28 = nn.Conv2d(256, 512, kernel_size=1, stride=2, bias=False)
        self.shortbn28 = nn.BatchNorm2d(512)
        self.shortcut28 = nn.Sequential(
            self.shortconv28,
            self.shortbn28,
            self.shortgate28
            )      
        
        #Basic block 15 (512, 512, 1)
        self.shortcut30 = nn.Sequential()
        self.conv30 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn30 = nn.BatchNorm2d(512)
        self.gate30 = GateLayer(512, 512, [1, -1, 1, 1])
        
        self.conv31 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn31 = nn.BatchNorm2d(512)
        self.gate31 = GateLayer(512, 512, [1, -1, 1, 1])
        
         #Basic block 16 (512, 512, 1)
        self.shortcut32 = nn.Sequential()
        self.conv32 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn32 = nn.BatchNorm2d(512)
        self.gate32 = GateLayer(512, 512, [1, -1, 1, 1])
        
        self.conv33 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn33 = nn.BatchNorm2d(512)
        self.gate33 = GateLayer(512, 512, [1, -1, 1, 1])
        
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, gate):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, gate))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.gate1(out)
        out = F.relu(out)
        
        # Basic Block 1
        shortcut = self.shortcut2(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.gate2(out)
        out = F.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.gate3(out)
        out = F.relu(out)

        out += shortcut
        
        # Basic Block 2
        shortcut = self.shortcut4(out)
        out = self.conv4(out)
        out = self.bn4(out)
        out = self.gate4(out)
        out = F.relu(out)
        
        out = self.conv5(out)
        out = self.bn5(out)
        out = self.gate5(out)
        out = F.relu(out)

        out += shortcut
        
        # Basic Block 3
        shortcut = self.shortcut6(out)
        out = self.conv6(out)
        out = self.bn6(out)
        out = self.gate6(out)
        out = F.relu(out)
        
        out = self.conv7(out)
        out = self.bn7(out)
        out = self.gate7(out)
        out = F.relu(out)

        out += shortcut
        
        # Basic Block 4
        shortcut = self.shortcut8(out)
        out = self.conv8(out)
        out = self.bn8(out)
        out = self.gate8(out)
        out = F.relu(out)
        
        out = self.conv9(out)
        out = self.bn9(out)
        out = self.gate9(out)
        out = F.relu(out)
        
        out += shortcut
        
        # Basic Block 5
        shortcut = self.shortcut10(out)
        out = self.conv10(out)
        out = self.bn10(out)
        out = self.gate10(out)
        out = F.relu(out)
        
        out = self.conv11(out)
        out = self.bn11(out)
        out = self.gate11(out)
        out = F.relu(out)

        out += shortcut
        
        # Basic Block 6
        shortcut = self.shortcut12(out)
        out = self.conv12(out)
        out = self.bn12(out)
        out = self.gate12(out)
        out = F.relu(out)
        
        out = self.conv13(out)
        out = self.bn13(out)
        out = self.gate13(out)
        out = F.relu(out)

        out += shortcut
        
        # Basic Block 7
        shortcut = self.shortcut14(out)
        out = self.conv14(out)
        out = self.bn14(out)
        out = self.gate14(out)
        out = F.relu(out)
        
        out = self.conv15(out)
        out = self.bn15(out)
        out = self.gate15(out)
        out = F.relu(out)
        
        out += shortcut
        
        # Basic Block 8
        shortcut = self.shortcut16(out)
        out = self.conv16(out)
        out = self.bn16(out)
        out = self.gate16(out)
        out = F.relu(out)
        
        out = self.conv17(out)
        out = self.bn17(out)
        out = self.gate17(out)
        out = F.relu(out)

        out += shortcut

        
        # Basic Block 9
        shortcut = self.shortcut18(out)
        out = self.conv18(out)
        out = self.bn18(out)
        out = self.gate18(out)
        out = F.relu(out)

        out = self.conv19(out)
        out = self.bn19(out)
        out = self.gate19(out)
        out = F.relu(out)

        out += shortcut
        
        # Basic Block 10
        shortcut = self.shortcut20(out)
        out = self.conv20(out)
        out = self.bn20(out)
        out = self.gate20(out)
        out = F.relu(out)
        
        out = self.conv21(out)
        out = self.bn21(out)
        out = self.gate21(out)
        out = F.relu(out)

        out += shortcut
        
        # Basic Block 6
        shortcut = self.shortcut22(out)
        out = self.conv22(out)
        out = self.bn22(out)
        out = self.gate22(out)
        out = F.relu(out)
        
        out = self.conv23(out)
        out = self.bn23(out)
        out = self.gate23(out)
        out = F.relu(out)

        out += shortcut
        
        # Basic Block 7
        shortcut = self.shortcut24(out)
        out = self.conv24(out)
        out = self.bn24(out)
        out = self.gate24(out)
        out = F.relu(out)
        
        out = self.conv25(out)
        out = self.bn25(out)
        out = self.gate25(out)
        out = F.relu(out)

        out += shortcut
        
        # Basic Block 8
        shortcut = self.shortcut26(out)
        out = self.conv26(out)
        out = self.bn26(out)
        out = self.gate26(out)
        out = F.relu(out)
        
        out = self.conv27(out)
        out = self.bn27(out)
        out = self.gate27(out)
        out = F.relu(out)

        out += shortcut

        # Basic Block 9
        shortcut = self.shortcut28(out)
        out = self.conv28(out)
        out = self.bn28(out)
        out = self.gate28(out)
        out = F.relu(out)
        
        out = self.conv29(out)
        out = self.bn29(out)
        out = self.gate29(out)
        out = F.relu(out)

        out += shortcut
        
        # Basic Block 10
        shortcut = self.shortcut30(out)
        out = self.conv30(out)
        out = self.bn30(out)
        out = self.gate30(out)
        out = F.relu(out)
        
        out = self.conv31(out)
        out = self.bn31(out)
        out = self.gate31(out)
        out = F.relu(out)

        out += shortcut        
        
        # Basic Block 11
        shortcut = self.shortcut32(out)
        out = self.conv32(out)
        out = self.bn32(out)
        out = self.gate32(out)
        out = F.relu(out)
        
        out = self.conv33(out)
        out = self.bn33(out)
        out = self.gate33(out)
        out = F.relu(out)

        out += shortcut 
        
        # fc
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out

