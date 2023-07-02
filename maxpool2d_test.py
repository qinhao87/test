import torch as t
import torch.nn as nn
from torch.autograd import Variable

m=nn.MaxPool1d(3,stride=1)
input=Variable(t.randn(2,4,5))
print(input)
print(m(input))

m=nn.MaxPool2d(3,stride=2)
input=Variable(t.randn(2,4,5))
print(input)
print(m(input))