from torch.nn import *


class AudioClassificationModel(Module):
    def __init__(self, num_classes):
        super().__init__()
        conv_layers = []
        self.conv1 = Conv2d(2, 8, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
        self.relu1 = ReLU()
        self.bn1 = BatchNorm2d(8)
        conv_layers += [self.conv1, self.relu1, self.bn1]

        self.conv2 = Conv2d(8, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu2 = ReLU()
        self.bn2 = BatchNorm2d(16)
        conv_layers += [self.conv2, self.relu2, self.bn2]

        self.conv3 = Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu3 = ReLU()
        self.bn3 = BatchNorm2d(32)
        conv_layers += [self.conv3, self.relu3, self.bn3]

        self.conv4 = Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu4 = ReLU()
        self.bn4 = BatchNorm2d(64)
        conv_layers += [self.conv4, self.relu4, self.bn4]

        self.ap = AdaptiveAvgPool2d(output_size=1)
        self.classification = Linear(in_features=64, out_features=num_classes)

        self.conv = Sequential(*conv_layers)  # *List："解引用"list,conv_layers是[[],[]]形式的

    def forward(self, x):
        x = self.conv(x)

        # flatten
        x = self.ap(x)
        x = x.view(x.shape[0], -1)

        x = self.classification(x)

        return x
