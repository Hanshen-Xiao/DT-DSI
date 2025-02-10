import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from functools import partial


__all__ = ['ResNet', 'WRN_40_4', 'WRN_16_4']

active = F.elu
# Notice nn.ReLU() is used in shortcut as F.relu is not a subclass of nn.Module
# no activation needed in shortcut in the modified version
norm_fn = partial(nn.GroupNorm, num_groups=16, affine=False)

def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, StdConv2d):
        init.kaiming_normal_(m.weight)

class StdConv2d(nn.Conv2d):
  def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
    super().__init__(in_channels, out_channels, kernel_size, **kwargs)
  def forward(self, x):        
    weight = self.weight
    weight_mean = weight.mean(dim=(1,2,3), keepdim=True)
    std = weight.std(dim=(1,2,3), keepdim=True) + 1e-6
    weight = (weight - weight_mean)/ std / (weight.numel() / weight.size(0))**0.5
    return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
  
class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class WideBasicBlock(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, option='B'):
        super(WideBasicBlock, self).__init__()

        self.conv1 = StdConv2d(in_planes, self.expansion * planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.n1 = norm_fn(num_channels= self.expansion * planes)
        self.conv2 = StdConv2d(self.expansion * planes, self.expansion * planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.n2 = norm_fn(num_channels= self.expansion * planes)
        self.n3 = norm_fn(num_channels= self.expansion * planes)
    

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     StdConv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     norm_fn(num_channels= self.expansion * planes),
                )

    def forward(self, x):

        out = self.n1(active(self.conv1(x)))
        out = self.n2(self.conv2(out))
        out += self.shortcut(x)
        out = self.n3(active(out))

        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, width=4, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16
        block.expansion = width

        self.conv1 = StdConv2d(3, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.n1 = norm_fn(num_channels=self.in_planes)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(self.in_planes*4, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.n1(active(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.adaptive_avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = (out - out.mean(dim=1, keepdim=True)) / (out.std(dim=1, keepdim=True)+ 1e-6)
        out = self.linear(out)
        return out

def WRN_40_4():
    return ResNet(WideBasicBlock, [6, 6, 6], width=4)

def WRN_16_4():
    return ResNet(WideBasicBlock, [2, 2, 2], width=4)