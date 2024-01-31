import torch.nn as nn
import torch.functional as F
import torch.cuda


if torch.cuda.is_available():
    device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")


class DFnet(nn.Module):
    def __init__(self, in_channel, num_classes):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels=in_channel, out_channels=32, kernel_size=8, stride=1, padding=4),
            nn.BatchNorm1d(num_features=32),
            nn.ELU(alpha=0.1),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=8, stride=1, padding=3),
            nn.BatchNorm1d(num_features=32),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=8, stride=2, padding=3),  # 2500
            nn.Dropout(p=0.1)
        )

        self.block2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=8, stride=1, padding=4),
            nn.BatchNorm1d(num_features=64),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=8, stride=1, padding=3),
            # 2500
            nn.BatchNorm1d(num_features=64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=8, stride=2, padding=3),  # 1250
            nn.Dropout(0.1)
        )

        self.block3 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=8, stride=1, padding=4),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=8, stride=1, padding=3),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=8, stride=2),  # 622
            nn.Dropout(0.1)
        )

        self.block4 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=8, stride=1, padding=4),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU(),
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=8, stride=1, padding=3),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=8, stride=2),  # 311
            nn.Dropout(0.1)
        )

        self.FCblock = nn.Sequential(
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(in_features=256 * 308, out_features=512),
            nn.BatchNorm1d(num_features=512),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(in_features=512, out_features=512),
            nn.BatchNorm1d(num_features=512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.predblock = nn.Linear(in_features=512, out_features=num_classes)

    def forward(self, x, features=False):
        y_block1 = self.block1(x)
        y_block2 = self.block2(y_block1)
        y_block3 = self.block3(y_block2)
        y_block4 = self.block4(y_block3)
        y_flatten = self.FCblock(y_block4)
        if features:
            y_flatten = y_flatten / y_flatten.norm()
            return y_flatten
        else:
            y_pred = self.predblock(y_flatten)
            return y_pred


# --------------------------------------------------- #
# ------------ResNet 50 1D 模型---------------------- #
# --------------------------------------------------- #

class BottleNeck(nn.Module):
    def __init__(self, in_channel, med_channel, out_channel, downsample=False):
        super(BottleNeck, self).__init__()

        # 降采样 = pooling池化操作 ， 在此1维实现中，通过调控第一个卷积核的步长，实现降采样
        if downsample:
            self.stride = 2
        else:
            self.stride = 1

        self.layer = nn.Sequential(
            nn.Conv1d(in_channels=in_channel, out_channels=med_channel, kernel_size=1, stride=self.stride),
            nn.BatchNorm1d(num_features=med_channel),
            nn.ReLU(),
            nn.Conv1d(in_channels=med_channel, out_channels=med_channel, kernel_size=3, padding=1),
            nn.BatchNorm1d(num_features=med_channel),
            nn.ReLU(),
            nn.Conv1d(in_channels=med_channel, out_channels=out_channel, kernel_size=1),
            nn.BatchNorm1d(num_features=out_channel),
            nn.ReLU(),
        )

        # 为了保证输入、输出通道数一致，以便于残差块的直接相加
        # 此处步长也受到降采样的控制， 因为原始数据也要和降采样后的输出大小一致
        if in_channel != out_channel:  # 输入通道数 != 输出通道数 -- 增加一层卷积，卷积核大小为1
            self.res_layer = nn.Conv1d(in_channels=in_channel, out_channels=out_channel, kernel_size=1,
                                       stride=self.stride)
        else:
            self.res_layer = None

    def forward(self, x):
        if self.res_layer is not None:
            residual = self.res_layer(x)
        else:
            residual = x
        return self.layer(x) + residual


class ResNet50_1D(nn.Module):
    def __init__(self, in_channel, num_classes):
        super(ResNet50_1D, self).__init__()
        self.features = nn.Sequential(

            # conv1
            nn.Conv1d(in_channels=in_channel, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            # conv2_x
            BottleNeck(64, 64, 256, False),
            BottleNeck(256, 64, 256, False),
            BottleNeck(256, 64, 256, False),
            # conv3_x
            BottleNeck(256, 128, 512, True),
            BottleNeck(512, 128, 512, False),
            BottleNeck(512, 128, 512, False),
            BottleNeck(512, 128, 512, False),
            # conv4_x
            BottleNeck(512, 256, 1024, True),
            BottleNeck(1024, 256, 1024, False),
            BottleNeck(1024, 256, 1024, False),
            BottleNeck(1024, 256, 1024, False),
            BottleNeck(1024, 256, 1024, False),
            BottleNeck(1024, 256, 1024, False),
            # conv5_x
            BottleNeck(1024, 512, 2048, True),
            BottleNeck(2048, 512, 2048, False),
            BottleNeck(2048, 512, 2048, False),

            torch.nn.AdaptiveAvgPool1d(1)
        )
        self.classifer = torch.nn.Sequential(
            torch.nn.Linear(in_features=2048, out_features=num_classes)
        )

    def forward(self, x):
        x_features = self.features(x)
        x_features = x_features.view(-1, 2048)
        y_pred = self.classifer(x_features)
        return y_pred
