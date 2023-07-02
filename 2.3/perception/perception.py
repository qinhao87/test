import torch
from torch import nn
import ipdb
import torch.nn.functional as F
class linear(nn.Module):
    def __init__(self, in_dem, out_dim):
        super(linear, self).__init__()
        self.w = nn.Parameter(torch.randn(in_dem, out_dim))
        self.b = nn.Parameter(torch.randn(out_dim))

    def forward(self, x):
        x = x.matmul(self.w)
        y = self.b.expand_as(x) + x
        return y


class Perception(nn.Module):
    def __init__(self,in_dim,h_dim,out_dim):
        super(Perception,self).__init__()
        self.layer1=linear(in_dim,h_dim)
        self.layer2=linear(h_dim,out_dim)

    def forward(self,x):
        x=self.layer1(x)
        y=torch.sigmoid(x)
        y=self.layer2(y)
        y=torch.sigmoid(y)
        return y

class Sequential_Perception(nn.Module):
    def __init__(self,in_dim,h_dim,out_dim):
        super(Sequential_Perception,self).__init__()
        self.layer=nn.Sequential(linear(in_dim,h_dim),nn.Sigmoid(),
                                 linear(h_dim,out_dim),nn.Sigmoid())

    def forward(self,x):
        y=self.layer(x)
        return y


# data=torch.randn(2,2)
# print(data)
# # ipdb.set_trace()
# lin=Sequential_Perception(2,4,2)
# print(lin)
# for name,parameter in lin.named_parameters():
#     print(name,parameter)
# res=lin(data)
# print(res)
model=Sequential_Perception(100,1000,10).cuda()
print(model)
input=torch.randn(100).cuda()
output=model(input).cpu()
print(output.unsqueeze_(0),output.shape)
# ipdb.set_trace()
label=torch.Tensor([1]).long()
print(label)
criterion=nn.CrossEntropyLoss()
loss_nn=criterion(output,label)
print(loss_nn)
loss_function=F.cross_entropy(output,label)
print(loss_function)
