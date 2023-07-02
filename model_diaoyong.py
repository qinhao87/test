import torch
import pickle

filename='E:/py_pro/simple-faster-rcnn-pytorch-master/checkpoints/archive/data.pkl'
f=open(filename,'rb')
#with open(filename,'rb') as f:
modeldata=pickle.load(f)
print(modeldata)
# modeldata=torch.load(filename)
# print(modeldata)