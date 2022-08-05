from itertools import chain
import layers
import torch
import pprint
from torchvision.models import resnet50
import torchvision.transforms as transforms

Conv2d = torch.nn.Conv2d
BatchNorm2d = torch.nn.BatchNorm2d
LeakyReLU = torch.nn.LeakyReLU
MaxPool2d = torch.nn.MaxPool2d
Dropout = torch.nn.Dropout
AvgPool2d = torch.nn.AvgPool2d
Linear = torch.nn.Linear
Flatten = torch.nn.Flatten    

class Backbone_Module(torch.nn.Module):
    def __init__(self):
        super(Backbone_Module, self).__init__()
        self.conv_1 = torch.nn.Sequential(
            Conv2d(in_channels = 3, out_channels = 64, kernel_size=7, stride=2,  padding = 3),
            BatchNorm2d(64),
            LeakyReLU(),
        )
        self.max_1 = MaxPool2d(kernel_size = 2, stride = 2)
        self.conv_2 = torch.nn.Sequential(
            Conv2d(in_channels = 64, out_channels = 192, kernel_size = 3, stride = 1, padding = 1),
            BatchNorm2d(192),
            LeakyReLU(),
        )
        self.max_2 = MaxPool2d(kernel_size = 2, stride = 2)
        self.conv_3 = Conv2d(in_channels = 192, out_channels = 128,  kernel_size = 1, stride = 1, padding = 0)
        self.conv_4 = torch.nn.Sequential(
            Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3, stride = 1, padding = 1),
            BatchNorm2d(256),
            LeakyReLU(),
        )
        self.conv_5 = Conv2d(in_channels = 256, out_channels = 256,  kernel_size = 1, stride = 1, padding = 0)
        self.conv_6 = torch.nn.Sequential(
            Conv2d(in_channels = 256, out_channels = 512, kernel_size = 3, stride = 1, padding = 1),
            BatchNorm2d(512),
            LeakyReLU(),
        )
        self.max_6 = MaxPool2d(kernel_size = 2, stride = 2)
        chain_layers = []
        for _ in range(4):
            for l in [(1, 512, 256, 1, 0), (3, 256, 512, 1, 1)]:
                chain_layers.extend([
                                Conv2d(in_channels = l[1], out_channels = l[2], 
                                kernel_size = l[0], stride = l[3], padding = l[4]),
                                BatchNorm2d(l[2]),
                                LeakyReLU(),
                            ])
        self.chain_7 = torch.nn.Sequential(*chain_layers)
        self.conv_8 = Conv2d(in_channels = 512, out_channels = 512,  kernel_size = 1, stride = 1, padding = 0)        
        self.conv_9 = torch.nn.Sequential(
            Conv2d(in_channels = 512, out_channels = 1024, kernel_size = 3, stride = 1, padding = 1),
            BatchNorm2d(1024),
            LeakyReLU(),
        )
        self.max_9 = MaxPool2d(kernel_size = 2, stride = 2)
        chain_layers = []
        for _ in range(2):
            for l in [(1, 1024, 512, 1, 0), (3, 512, 1024, 1, 1)]:
                chain_layers.extend([
                                Conv2d(in_channels = l[1], out_channels = l[2], 
                                kernel_size = l[0], stride = l[3], padding = l[4]),
                                BatchNorm2d(l[2]),
                                LeakyReLU(),
                            ])
        self.chain_10 = torch.nn.Sequential(*chain_layers)
        self.conv_11 = torch.nn.Sequential(
            Conv2d(in_channels = 1024, out_channels = 1024, kernel_size = 3, stride = 1, padding = 1),
            BatchNorm2d(1024),
            LeakyReLU(),
        )
        self.conv_12 = torch.nn.Sequential(
            Conv2d(in_channels = 1024, out_channels = 1024, kernel_size = 3, stride = 2, padding = 1),
            BatchNorm2d(1024),
            LeakyReLU(),
        )
        self.conv_13 = torch.nn.Sequential(
            Conv2d(in_channels = 1024, out_channels = 1024, kernel_size = 3, stride = 1, padding = 1),
            BatchNorm2d(1024),
            LeakyReLU(),
        )
        self.conv_14 = torch.nn.Sequential(
            Conv2d(in_channels = 1024, out_channels = 1024, kernel_size = 3, stride = 1, padding = 1),
            BatchNorm2d(1024),
            LeakyReLU(),
        )

    def forward(self, x):
        x = self.max_1(self.conv_1(x))
        x = self.max_2(self.conv_2(x))
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.conv_5(x)
        x = self.max_6(self.conv_6(x))
        x = self.chain_7(x)
        x = self.conv_8(x)
        x = self.max_9(self.conv_9(x))
        x = self.chain_10(x)
        x = self.conv_11(x)
        x = self.conv_12(x)
        x = self.conv_13(x)
        x = self.conv_14(x)
        return x


class Head_Module(torch.nn.Module):
    def __init__(self, pretrained, S: int = 7, B: int = 2, C: int = 20):
        super(Head_Module, self).__init__()
        self.S = S
        self.B = B
        self.C = C
        lin_input = 2048 if pretrained else 1024 
        self.linear = Linear(lin_input, S * S * (B * 5 + C))
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p = 0.3)
    def forward(self, x):
        x = torch.nn.Sequential(
            AvgPool2d(x.size(-1)),
            Flatten(),
            self.linear, 
            self.relu, 
            self.dropout
        )(x)
        return x.view(-1, self.S, self.S, (self.B * 5 + self.C))

class ToTensor(torch.nn.Module):
    def __init__(self):
        super(ToTensor, self).__init__()

    def forward(self, x):
        x = x.type(torch.FloatTensor)
        x = x.permute(0, 3, 1, 2) / 255.0
        return x

def YOLO_Network(use_pretrained: bool = True) -> torch.nn.Module:
    def Resnet_Module() -> torch.nn.Module:
        pretrained_resnet50 = resnet50(pretrained = True)
        # Freeze the outer model
        for param in pretrained_resnet50.parameters():
            param.requires_grad = False
        # get all layers except fully connected layer (last part)
        list_modules = [m for m in pretrained_resnet50.children()][:-1]
        pretrained_model = torch.nn.Sequential(*list_modules)
        return pretrained_model
    # WHERE ALL THE ACTION TAKES PLACE
    preprocessor = torch.nn.Sequential(
       ToTensor(),
       transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    )
    if use_pretrained:
        x = torch.nn.Sequential(
            preprocessor,
            Resnet_Module(),
            Head_Module(use_pretrained)
        )
    else:
        x = torch.nn.Sequential(
            preprocessor,
            Backbone_Module(),
            Head_Module(use_pretrained)
        )
    return x

if __name__ == "__main__":
    yolo = YOLO_Network(use_pretrained = False)
    in_ = torch.randint(0, 255, (1, 448, 448, 3))
    out_ = yolo(in_)
    print(out_.shape)
    # print(yolo.summary())
    # for layer in yolo.layers:
    #     print(
    #         (" ".join(layer.name.split("_")[:-1])) +
    #             f"\tOUTPUT: {layer.output_shape}" + 
    #             f"\tPARAMS: {layer.count_params()}")



