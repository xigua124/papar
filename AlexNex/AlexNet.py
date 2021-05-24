from torch import nn
import torchvision.models
import torch


class AlexNet(nn.Module):

    def __init__(self, num_class=1000):
        super().__init__()
        self.num_class = num_class
        self.feature = nn.Sequential(
            # ConV1
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=11, stride=4, padding=2),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # ConV2
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=5, padding=2),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # ConV3
            nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3, padding=1),
            nn.PReLU(),

            # ConV4
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
            nn.PReLU(),

            # ConV5
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),

        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.FC1 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(in_features=256*6*6, out_features=4096),
            nn.PReLU()
        )
        self.FC2 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(in_features=4096, out_features=self.num_class),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.feature(x)
        print(x.shape)
        x = self.avgpool(x)
        print(x.shape)
        x = x.reshape((-1, 256*6*6))
        x = self.FC1(x)
        x = self.FC2(x)
        return x


if __name__ == '__main__':
    net = AlexNet(2)
    x = torch.randn((1, 3, 224, 224))
    x = net(x)
    print(x)
