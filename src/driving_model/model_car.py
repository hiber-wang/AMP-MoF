import torch
import torch.nn as nn
from torchvision import models

class EpochSpeed(nn.Module):
    def __init__(self):
        super(EpochSpeed, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer5 = nn.Sequential(
            nn.Linear(50176, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.ReLU()
        )
    
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.reshape(out.size(0), -1)
        out = self.layer5(out)
        return out
    


class EpochSteer(nn.Module):
    def __init__(self):
        super(EpochSteer, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer5 = nn.Sequential(
            nn.Linear(50176, 2048),
            nn.Tanh(),
            nn.Linear(2048, 1024),
            nn.Tanh(),
            nn.Linear(1024, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Tanh()
        )
    
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.reshape(out.size(0), -1)
        out = self.layer5(out)
        return out


class Resnet101Steer(nn.Module):
    def __init__(self, pretrained=False):
        super(Resnet101Steer, self).__init__()
        self.model = models.resnet101(pretrained=pretrained)
        if pretrained:
            for parma in self.model.parameters():
                parma.requires_grad = False
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
                            nn.BatchNorm1d(num_ftrs, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                            nn.Dropout(p=0.5),
                            nn.Linear(in_features=2048, out_features=256, bias=True),
                            nn.Tanh()
                            )
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Tanh()
        )

    def forward(self, x):
        out = self.model(x)
        out = self.fc(out)
        return out        


class Resnet101Speed(nn.Module):
    def __init__(self, pretrained=False):
        super(Resnet101Speed, self).__init__()
        self.model = models.resnet101(pretrained=pretrained)
        if pretrained:
            for parma in self.model.parameters():
                parma.requires_grad = False
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
                            nn.BatchNorm1d(num_ftrs, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                            nn.Dropout(p=0.25),
                            nn.Linear(in_features=2048, out_features=256, bias=True),
                            nn.LeakyReLU()
                            )
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.model(x)
        out = self.fc(out)
        return out        


class Vgg16Steer(nn.Module):
    def __init__(self, pretrained=True):
        super(Vgg16Steer, self).__init__()
        self.model = models.vgg16(pretrained=pretrained)
        if pretrained:
            for parma in self.model.parameters():
                parma.requires_grad = False
        self.conv_new = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
        )

        self.model.classifier = nn.Sequential(
            nn.BatchNorm1d(512*7*7),
            nn.Dropout(p=0.5),
            nn.Linear(512*7*7, 1024),
            nn.Tanh(),
            nn.Linear(1024, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh()
        )

        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )
        # self.fc2 = nn.Linear(2, 1)
    def forward(self, x):
        out = self.model(x)
        out = self.fc(out)
        return out        
    


class Vgg16Speed(nn.Module):
    def __init__(self, pretrained=True):
        super(Vgg16Speed, self).__init__()
        self.model = models.vgg16(pretrained=pretrained)
        if pretrained:
            for parma in self.model.parameters():
                parma.requires_grad = False
        self.conv_new = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        self.model.classifier = nn.Sequential(
            nn.BatchNorm1d(512*7*7),
            nn.Linear(512*7*7, 256)
        )

        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1),
            nn.ReLU(),
        )
    def forward(self, x):
        out = self.model(x)
        out = self.fc(out)
        return out 
    

if __name__ == "__main__":
    model = Vgg16Speed()
    num_of_parameter = sum(p.numel() for p in model.parameters())
    print(num_of_parameter)
