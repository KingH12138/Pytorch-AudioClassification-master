from torch import nn
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152


class Resnet18(nn.Module):
    def __init__(self, num_classes, pretrain=False):
        super(Resnet18, self).__init__()
        self.resnet18 = resnet18(pretrained=pretrain)
        self.resnet18.fc = nn.Linear(in_features=512, out_features=num_classes, bias=True)

    def forward(self, x):
        return self.resnet18(x)


class Resnet34(nn.Module):
    def __init__(self, num_classes, pretrain=False):
        super(Resnet34, self).__init__()
        self.resnet34 = resnet34(pretrained=pretrain)
        self.resnet34.fc = nn.Linear(in_features=512, out_features=num_classes, bias=True)

    def forward(self, x):
        return self.resnet34(x)


class Resnet50(nn.Module):
    def __init__(self, num_classes, pretrain=False):
        super(Resnet50, self).__init__()
        self.resnet50 = resnet50(pretrained=pretrain)
        self.resnet50.fc = nn.Linear(in_features=2048, out_features=num_classes, bias=True)

    def forward(self, x):
        return self.resnet50(x)


class Resnet101(nn.Module):
    def __init__(self, num_classes, pretrain=False):
        super(Resnet101, self).__init__()
        self.resnet101 = resnet101(pretrained=pretrain)
        self.resnet101.fc = nn.Linear(in_features=2048, out_features=num_classes, bias=True)

    def forward(self, x):
        return self.resnet101(x)


class Resnet152(nn.Module):
    def __init__(self, num_classes, pretrain=False):
        super(Resnet152, self).__init__()
        self.resnet152 = resnet152(pretrained=pretrain)
        self.resnet152.fc = nn.Linear(in_features=2048, out_features=num_classes, bias=True)

    def forward(self, x):
        return self.resnet152(x)
