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

        self.predblock = nn.Sequential(
            nn.Linear(in_features=512, out_features=num_classes),
            # nn.Softmax(dim=-1)
        )

    def forward(self, x):
        y_block1 = self.block1(x)
        y_block2 = self.block2(y_block1)
        y_block3 = self.block3(y_block2)
        y_block4 = self.block4(y_block3)
        y_flatten = self.FCblock(y_block4)
        y_pred = self.predblock(y_flatten)
        return y_pred
