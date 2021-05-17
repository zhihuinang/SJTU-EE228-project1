import torch.nn as nn
import torch.nn.functional as F

class base_block(nn.Module):
    def __init__(self,in_ch,out_ch,stride=1):
        super(base_block, self).__init__()
        self.conv1 = nn.Conv2d(
            in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.Res = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.Res = nn.Sequential(
                nn.Conv2d(in_ch, out_ch,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)),inplace=True)
        out = self.bn2(self.conv2(out))
        out += self.Res(x)
        out = F.relu(out,inplace=True)
        return out

class bottleneck(nn.Module):
    def __init__(self,in_ch,out_ch,stride=1):
        super(bottleneck,self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.conv3 = nn.Conv2d(out_ch, 4*out_ch, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(4*out_ch)

        self.Res = nn.Sequential()
        if stride != 1 or in_ch != 4*256:
            self.Res = nn.Sequential(
                nn.Conv2d(in_ch, 4*out_ch,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(4*out_ch)
            )
    def forward(self,x):
        out = F.relu(self.bn1(self.conv1(x)),inplace=True)
        out = F.relu(self.bn2(self.conv2(out)),inplace=True)
        out = self.bn3(self.conv3(out))
        out += self.Res(x)
        out = F.relu(out,inplace=True)
        return out


class ResNet18(nn.Module):
    def __init__(self,in_ch,class_num):
        super(ResNet18,self).__init__()
        num_blocks = [2,2,2,2]
        self.in_ch = 64
        self.conv1 = nn.Conv2d(in_ch,64,kernel_size=5,stride=1,padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        #self.pool1 = nn.MaxPool2d(kernel_size=3,stride=2)
        self.layer1 = self._make_layer(64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512, class_num)


    def _make_layer(self, planes, num_blocks, stride):
            strides = [stride] + [1]*(num_blocks-1)
            layers = []
            for stride in strides:
                layers.append(base_block(self.in_ch, planes, stride))
                self.in_ch = planes
            return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)),inplace=True)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class ResNet50(nn.Module):
    def __init__(self,in_ch,class_num):
        super(ResNet50,self).__init__()
        num_blocks = [3,4,6,3]
        self.in_ch = 64
        self.conv1 = nn.Conv2d(in_ch,64,kernel_size=5,stride=1,padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        #self.pool1 = nn.MaxPool2d(kernel_size=3,stride=2)
        self.layer1 = self._make_layer(64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(512, num_blocks[3], stride=2)
        self.linear = nn.Linear(2048, class_num)


    def _make_layer(self, planes, num_blocks, stride):
            strides = [stride] + [1]*(num_blocks-1)
            layers = []
            for stride in strides:
                layers.append(bottleneck(self.in_ch, planes, stride))
                self.in_ch = planes*4
            return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)),inplace=True)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

