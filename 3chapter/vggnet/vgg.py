import torch
from torch import nn

class VGG(nn.Module):
    def __init__(self,num_classes=1000):
        super(VGG,self).__init__()
        layers=[]
        in_dim=3
        out_dim=64
        for i in range(13):
            # import ipdb;ipdb.set_trace()
            layers+=[nn.Conv2d(in_dim,out_dim,3,1,1),nn.ReLU(inplace=True)]
            in_dim=out_dim
            if i == 1 or i == 3 or i==6 or i==9 or i==12:
                layers+=[nn.MaxPool2d(kernel_size=2,stride=2)]
                if i!=9:
                    out_dim*=2

        self.feature= nn.Sequential(*layers)
        self.classifier=nn.Sequential(
            nn.Linear(512*7*7,4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096,4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096,num_classes)
        )
    def forward(self,x):
        x=self.feature(x)
        x=x.view(x.size(0),-1)
        x=self.classifier(x)
        return x
