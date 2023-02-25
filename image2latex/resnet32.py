import torch
import torch.nn as nn
from torchsummary import summary


import torch.nn as nn


class ResnetBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, padding='same'):
        super(ResnetBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=padding,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    self.expansion * out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * out_channels),
            )

    def forward(self, x):
        out = nn.ReLU()(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = nn.ReLU()(out)
        return out


class ResnetBackBone(nn.Module):
    def __init__(self):

    
        super(ResnetBackBone, self).__init__()
    
        # Conv1_x
        self.conv1_1 = nn.Conv2d(
            1, 32, stride=(1, 1), padding=(1, 1), kernel_size=(3, 3)
        )
        self.conv1_2 = nn.Conv2d(
            32, 64, stride=(1, 1), padding=(1, 1), kernel_size=(3, 3)
        )

        # Conv2_x
        self.conv2_pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0)
        self.conv2_resnet1 = ResnetBlock(in_channels=64, out_channels=128)
            # self.conv2_resnet2 = ResnetBlock(in_channels=128, out_channels=128)
        self.conv2_1 = nn.Conv2d(
            128, 128, stride=(1, 1), padding=(1, 1), kernel_size=(3, 3)
        )

        # Conv3_x
        self.conv3_pool = nn.MaxPool2d(2, stride = 2, padding = 0)
        self.conv3_1 = ResnetBlock(in_channels=128, out_channels=256)
        self.conv3_2 = ResnetBlock(in_channels=256, out_channels=256)

        # Conv4_x
        self.conv4_pool = nn.MaxPool2d(2, stride=(2,1), padding=(0,1))
        self.conv4_1 = ResnetBlock(in_channels=256, out_channels=512)
        self.conv4_2 = ResnetBlock(in_channels=512, out_channels=512)
        self.conv4_3 = ResnetBlock(in_channels=512, out_channels=512)
        self.conv4_4 = ResnetBlock(in_channels=512, out_channels=512)
        self.conv4_5 = ResnetBlock(in_channels=512, out_channels=512)
        self.conv4_11 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)

        # Conv5_x
        self.conv5_1 = ResnetBlock(in_channels=512, out_channels=512)
        self.conv5_2 = ResnetBlock(in_channels=512, out_channels=512)
        self.conv5_3 = ResnetBlock(in_channels=512, out_channels=512)
        self.conv5_7 = nn.Conv2d(512, 512, kernel_size=2, stride=(2,1), padding=(0,1))
        self.conv5_8 = nn.Conv2d(512, 512, kernel_size=2, stride=(1,1), padding = 0)




    def forward(self, x):
        out = self.conv1_1(x)
        out = self.conv1_2(out)

        out = self.conv2_pool(out)
        out = self.conv2_resnet1(out)
        out = self.conv2_1(out)

        out = self.conv3_pool(out)
        out = self.conv3_1(out)
        out = self.conv3_2(out)

        out = self.conv4_pool(out)
        out = self.conv4_1(out)
        out = self.conv4_2(out)
        out = self.conv4_3(out)
        out = self.conv4_4(out)
        out = self.conv4_5(out)
        out = self.conv4_11(out)

        out = self.conv5_1(out)
        out = self.conv5_2(out)
        out = self.conv5_3(out)
        out = self.conv5_7(out)
        out = self.conv5_8(out)
        return out

def main():
    summary(ResnetBackBone().to('cpu'), (1, 64, 256), batch_size=1)

if __name__ == '__main__':
    main()