import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import linalg as LA




class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, HW, stride=1):
        super(BasicBlock, self).__init__()

        #self.kwargs = kwargs
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, nclass=10, scale=1, channels=3, **kwargs):
        super(ResNet, self).__init__()
        #self.kwargs = kwargs
        self.in_planes = int(64 * scale)
        self.channels = channels
        #print (self.in_planes)

        
        self.conv1 = nn.Conv2d(self.channels, int(64 * scale), kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(int(64*scale))

        self.layer1 = self._make_layer(block, int(64 * scale), num_blocks[0], HW=32, stride=1)
        self.layer2 = self._make_layer(block, int(128 * scale), num_blocks[1], HW=16, stride=2)
        self.layer3 = self._make_layer(block, int(256 * scale), num_blocks[2], HW=8, stride=2)
        self.layer4 = self._make_layer(block, int(512 * scale), num_blocks[3], HW=4, stride=2)
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.linear = nn.Linear(int(512 * scale) * block.expansion, nclass)

        self.multi_out = 0


            
            
    def _make_layer(self, block, planes, num_blocks, HW, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, HW, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)            
        out = self.layer4(out)
        
        out = self.gap(out)

        p = out.view(out.size(0), -1)

        out = self.linear(p)
        
        if (self.multi_out):
            return p, out
        else:
            return out


class ResNetTiny(nn.Module):
    def __init__(self, block, num_blocks, nclass=10, scale=1, channels=3, **kwargs):
        super(ResNetTiny, self).__init__()
        #self.kwargs = kwargs        
        self.in_planes = int(64 * scale)
        self.channels = channels

        self.conv1 = nn.Conv2d(self.channels, int(64 * scale), kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(int(64*scale))

        self.layer1 = self._make_layer(block, int(64*scale), num_blocks[0], HW=32, stride=1)
        self.layer2 = self._make_layer(block, int(128*scale), num_blocks[1], HW=16, stride=2)
        self.layer3 = self._make_layer(block, int(256*scale), num_blocks[2], HW=8, stride=2)
        self.layer4 = self._make_layer(block, int(512*scale), num_blocks[3], HW=4, stride=2)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(int(512*scale) * block.expansion, nclass)


        self.multi_out = 0


    def _make_layer(self, block, planes, num_blocks, HW, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, HW, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)            
        out = self.layer4(out)
        
        out = self.gap(out)

        p = out.view(out.size(0), -1)

        out = self.linear(p)
        
        if (self.multi_out):
            return p, out
        else:
            return out




def ResNet18(nclass, scale, channels, **kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], nclass, scale, channels, **kwargs)

def ResNet18Tiny(nclass, scale, channels, **kwargs):
    return ResNetTiny(BasicBlock, [2, 2, 2, 2], nclass, scale, channels, **kwargs)

def set_bn_eval(m):
    if isinstance(m, nn.modules.batchnorm._BatchNorm):
        m.eval()

def set_bn_train(m):
    if isinstance(m, nn.modules.batchnorm._BatchNorm):
        m.train()
