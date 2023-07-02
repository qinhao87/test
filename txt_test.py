# filename='qinhao.txt'
# with open(filename,'a',encoding='utf-8') as file_object:
#     a=input()
#     file_object.write('\n'+a+'\n')
# print("i love you")

import torch as t
import numpy as np
VOC_BBOX_LABEL_NAMES = (
    'fly',
    'bike',
    'bird',
    'boat',
    'pin',
    'bus',
    'c',
    'cat',
    'chair',
    'cow',
    'table',
    'dog',
    'horse',
    'moto',
    'p',
    'plant',
    'shep',
    'sofa',
    'train',
    'tv',
)
epoch=int(input())

#print(type(epoch))
f='pred_bbox.txt'
labels=t.Tensor([1,2,3,14]).tolist()
print(labels)
print(type(labels))
labels = [int(label) for label in labels]
# int_labels=map(int,labels)
# labels = [label for label in int_labels]

print(labels)
labels=[VOC_BBOX_LABEL_NAMES[label] for label in labels]
print(labels)
#print(labels.dtype)
# for label in labels:
#     c=VOC_BBOX_LABEL_NAMES[label]
#     c.append(c)
#
# print(c)
with open(f,'a') as f_obj:
#    labels=input()
    gt_bbox=input()
    pred_score=input()
    f_obj.write('labels={0},gt_bbox={1},pred_score={2}\n'.format(labels,gt_bbox,pred_score))
print(labels,gt_bbox,pred_score)