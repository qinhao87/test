# # import numpy as np
# # import json
# # from utils.config import opt
# # from data.dataset import TestDataset
# # import torch
# #
# # VOC_BBOX_LABEL_NAMES = (
# #     'fly',
# #     'bike',
# #     'bird',
# #     'boat',
# #     'pin',
# #     'bus',
# #     'c',
# #     'cat',
# #     'chair',
# #     'cow',
# #     'table',
# #     'dog',
# #     'horse',
# #     'moto',
# #     'p',
# #     'plant',
# #     'shep',
# #     'sofa',
# #     'train',
# #     'tv',
# # )
# #
# # numbers=[]
# #
# # testset=TestDataset(opt)
# # voc_data_name=testset.db.ids
# #
# # def inf(ii,pictrue_name,gt_labels,gt_difficults,gt_bboxes,pred_labels,pred_bboxes,pred_scores):
# #
# #     picture_name=pictrue_name[ii]
# #
# #     gt_labels=gt_labels.squeeze_(0).tolist()
# #     gt_labels = [int(gt_label) for gt_label in gt_labels]
# #     gt_labels = [VOC_BBOX_LABEL_NAMES[gt_label] for gt_label in gt_labels]
# #
# #     gt_difficults=gt_difficults
# #
# #     gt_bboxes=gt_bboxes
# #
# #     pred_labels=pred_labels[0].tolist()
# #     pred_labels=[int(pred_label) for pred_label in pred_labels]
# #     pred_labels=[VOC_BBOX_LABEL_NAMES[pred_label] for pred_label in pred_labels]
# #
# #     pred_bboxes=pred_bboxes
# #
# #     pred_scores=pred_scores
# #
# #     information=[{'picture_name':picture_name,'gt_labels':gt_labels,'gt_difficults':gt_difficults,
# #                 'gt_bboxes':gt_bboxes,'pred_labels':pred_labels,'pred_bboxes':pred_bboxes,'pred_scores':pred_scores}]
# #
# #
# #     numbers.append(information)
# #
# #
# #     filename='test.json'
# #
# #     with open(filename,'w') as f_obj:
# #         json.dump(numbers,f_obj)
# #
# #     # with open(filename) as f_obj:
# #     #     number=json.load(f_obj)
# #     #
# #     # print(number[0][0]['gt_labels'])
#
#
# # a=(0,)
# # print(a*4)
# # import numpy as np
# # a=np.array([[1,2,3,4],[5,100,7,8],[10,11,12,13]])
# # print(a.argmax(axis=1))
# # import xml.etree.ElementTree as ET
# #
# # tree= ET.parse('E:/data/test\Annotations/0_326.xml')
# # if tree:
# #     print('1')
# # import numpy as np
# # keep=[1,2,3,4]
# # a=5*np.ones((len(keep),))
# # print(a)
# # import numpy as np
# # # img=np.array([1,2,3,4,5])#不是列表，而是数组哦
# # # img=img[:,None]
# # #
# # # print(np.shape(img))
# # # # print(np.zeros((0,4)))
# # ratios=[0.5, 1, 2]
# # anchor_scales=[8, 16, 32]
# # anchor_base = np.zeros((len(ratios) * len(anchor_scales), 4),
# #                        dtype=np.float32)
# # print(anchor_base)
#
#
# # a={'0':[1,2,3,4],'1':[12,345,6]}
# # b={'0':[15,123],'3':[21345]}
# # e={'1':[2,3],'5':[7,2],'3':[63,12]}
# # list_info=[a,b,e,'txt']
# #
# # while len(list_info) !=2:
# #     for d in list_info[1].keys():
# #         if d in list_info[0].keys():
# #             list_info[0][d].extend(list_info[1][d])
# #         else:
# #             list_info[0][d]=list_info[1][d]
#
#
#
# #     del list_info[1]
# # print(list_info)
#
#
#
# # print(a+b)
# # a.update(b)
# # print(a)
# # c=a.keys()
# # # print(c)
# # for d in b.keys():
# #     if d in c:
# #         a[d].extend(b[d])
# #     else:
# #         a[d]=b[d]
# # print(a)
#
# # f=open('qinhao.txt',encoding='utf-8')
# # lines=f.readlines()
# # for line in lines:
# #     line=line.strip()
# #     print(type(line))
# # import numpy as np
# # print(np.zeros(10))
# # import matplotlib.pyplot as plt
# # # plt.bar(left=0,height=1)
# # # plt.show()
# # from matplotlib.font_manager import FontProperties
# # salary=[2500,3300,2700,5600,6700,5400,3100,3500]
# #
# # group=[1000,2000,3000,4000]
# # plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
# # plt.rcParams['axes.unicode_minus'] = False
# # plt.hist(salary,group,histtype='bar',rwidth=0.8)
# #
# # plt.legend()
# #
# # plt.xlabel('salary-group')
# #
# # plt.ylabel('salary')
# #
# # plt.title(u'直方图')
# # plt.show()
# # import matplotlib
# # import matplotlib.pyplot as plt
# # # matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']
# # # matplotlib.rcParams['axes.unicode_minus'] = False
# # # x_axis=np.array([0,1,3,4,5,6,7,8,9])
# # # y_axis=10
# # #
# # # plt.hist(x_axis,bins=y_axis,color='blue',histtype='bar',rwidth=0.5)
# # # plt.xlabel('score')
# # # plt.ylabel('num')
# # # plt.show()
# # # import numpy as np
# # # import matplotlib
# # # import matplotlib.pyplot as plt
# # #
# # # a=range(0,1,10)
# # # bins=np.array([0,1,2,3,4,5,6,7,8,9])
# # # for i in range(len(a)):
# # #     y_data=np.random.randint(0,25,10)
# # #     print(y_data)
# # #     plt.subplot(len(a),1,i+1)
# # #     plt.hist(y_data,bins=bins,color='blue',histtype='bar',rwidth=0.5)
# # #     plt.xlabel('score')
# # #     plt.ylabel('num')
# # #     plt.show()
# # from matplotlib import pyplot as plt
# # from matplotlib import font_manager
# #
# # # 设置中文字体
# # my_font = font_manager.FontProperties(fname="/usr/share/fonts/truetype/arphic/ukai.ttc")
# #
# # x = ["战狼2", "速度与激情8", "功夫瑜伽", "西游伏妖篇", "变形金刚5：最后的骑士", "摔跤吧！爸爸", "加勒比海盗5：死无对证", "金刚：骷髅岛", "极限特工：终极回归", "生化危机6：终章",
# #      "乘风破浪", "神偷奶爸3", "智取威虎山", "大闹天竺", "金刚狼3：殊死一战", "蜘蛛侠：英雄归来", "悟空传", "银河护卫队2", "情圣", "新木乃伊", ]
# #
# # y = [56.01, 26.94, 17.53, 16.49, 15.45, 12.96, 11.8, 11.61, 11.28, 11.12, 10.49, 10.3, 8.75, 7.55, 7.32, 6.99, 6.88,
# #      6.86, 6.58, 6.23]
# #
# # # 设置图形大小
# # plt.figure(figsize=(18, 10), dpi=80)
# #
# # # 绘制条形图
# # plt.bar(range(len(x)), y, width=0.3)  # width表示条形粗细
# # # 绘制条形图 （横向条形图）
# # # plt.barh(range(len(x)), y, height=0.3, color="orange")  # 横向条形图中height表示条形粗细
# #
# # # 设置x轴刻度
# # plt.xticks(range(len(x)), x, fontproperties=my_font, rotation=90)
# # # plt.yticks(range(len(x)), x, fontproperties=my_font)  # barh()绘制横向条形图时，设置的是y轴刻度
# #
# # # 保存图片
# # plt.savefig("./movie.png")
# #
# # plt.show()
#
#
# # import matplotlib
# # import matplotlib.pyplot as plt
# # import numpy as np
# # import os
# # # import matplotlib.pyplot as plt
# #
# # # 这两行代码解决 plt 中文显示的问题
# # plt.rcParams['font.sans-serif'] = ['SimHei']
# # plt.rcParams['axes.unicode_minus'] = False
# #
# # score = (0.1, 0.2, 0.3, 0.4, 0.5,0.6,0.7,0.8,0.9,1.0)
# #
# # list_dict=[{'0':np.array([1,0,0,0,0,0,0,0,0,0]),'17':np.array([1,32,24,56,7,13,45,21,24,55])},'4948.txt']
# # class_num=len(list_dict[0].keys())
# # i=1
# # for cla in list_dict[0].keys():
# #     num=list_dict[0][cla]
# #     # plt.figure()
# #
# #     plt.subplot(class_num,1,i)
# #     plt.title(cla)
# #     plt.tight_layout(2)
# #     # axes[i].bar(score,num,width=0.05)
# #     plt.bar(score,num,width=0.05)
# #
# #     i+=1
# #
# # pic_name=list_dict[1].split('.')[0]+'.jpg'
# # plt.suptitle('图片为:'+pic_name)
# #
# # if  not os.path.exists('./save_bar'):
# #     os.mkdir('./save_bar')
# # plt.savefig('./save_bar/'+pic_name)
# # plt.show()
#
#
# #解包*args和**kargs
#
# # def f(a,*args):
# #     print(a)
# #     print(*args)
# #
# # def h(a,*args,**kargs):
# #     print(a)
# #     print(args)
# #     print(kargs)
#
#
#
# # if __name__=='__main__':
# #     f(1,2,3,4,5)
# #     h(1,2,3,4,5,x=1,y=2)
#
#
# # def foo():
# #     return inner
# #
# # def inner(num):
# #     print(num+1)
# #
# # a=foo()
# #
# # a(1)
# # def outer(x):
# #     def inner():
# #         print(x)
# #
# #     return inner
# # closure = outer(1)
# # closure() # 1
# import numpy as np
# import matplotlib.pyplot as plt
# lim = 4
# width = 0.4
# x = np.random.normal(0, 1, 10000)  # 生成均值为0,方差为1的正太分布点10000个
# bins = np.arange(-lim, lim, width)  # 设置直方图的分布区间 start->end step
# 直方图会进行统计各个区间的数值
# frequency_each, _1, _2 = plt.hist(x,
#                                   bins,
#                                   color='fuchsia',
#                                   alpha=1,
#                                   density=True)  # alpha设置透明度，0为完全透明
# plt.xlim(-lim, lim)  # 设置x轴分布范围
# plt.show()


# data=[0.9800252, 0.9404478, 0.93769133, 0.8174536, 0.8065584, 0.7250692, 0.7075952, 0.62857795, 0.57928556,
#       0.5492622, 0.46876234, 0.39974642, 0.36333662, 0.3537278, 0.2810385,
#       0.19720289, 0.1871258, 0.095572166, 0.07395064, 0.07000938, 0.060837615, 0.8938845]
# bins=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1]
# frequency_each, _1, _2 = plt.hist(data,
#                                   bins,
#                                   color='blue',
#                                   alpha=1,
#                                   rwidth=0.5)
#                                   # density=True)  # alpha设置透明度，0为完全透明
# # plt.xlim(0, 1)  # 设置x轴分布范围
# plt.ylim(0,10)
# plt.show()
# import numpy as np
# img=np.array([[1,2],[3,4],[5,6]])
# print(img.shape)
# img=img[None]
# print(img.shape)
# img=np.array([[1,2],[3,4],[5,6]])
# img=img[:,None,:]
# print(img.shape)
# img=np.array([[1,2],[3,4],[5,6]])
# img=img[:,:,None]
# print(img.shape)
# import numpy as np
# bins=np.linspace(0,0.05,10)
# bins=list(bins)
# # bins=[ x for x in range(0,0.51,0.01)]
# print(bins)
# planets = [
#     ('Mercury',2440,5.43,0.395),
#     ('Venus',6052,5.24,0.723),
#     ('Earth',6378,5.52,1.000),
#     ('Mars',3396,3.93,1.530)
# ]
# size = lambda  planet: planet[1]
# planets.sort(key=size,reverse=True)
# print(planets)
#
# import  numpy as np
# # gt_pre_iou=np.array([[1,2,3,4],[5,6,7,8]])
# # # print(type(gt_pre_iou))
# # iou_where=list(np.where(gt_pre_iou[0]>9)[0])
# # print(iou_where)
# # assert len(iou_where) is 0
# # if len(iou_where) == 0:
# #     print(True)
# a=list(range(20))
# a=[str(i) for i in a]
# cla_dict={}
# for k in a:
#     cla_dict[k]=[]
# cla_dict['0'].append(1)
# print(cla_dict)

# import argparse
#
# def main():
#     parser = argparse.ArgumentParser(description="Demo of argparse")
#     parser.add_argument('-n','--name', default=' Li ')
#     parser.add_argument('-y','--year', default='20')
#     args = parser.parse_args()
#     print(args)
#     name = args.name
#     year = args.year
#     print('Hello {}  {}'.format(name,year))
#
# if __name__ == '__main__':
#     main()
# import  argparse
#
# parser = argparse.ArgumentParser(description ='test of argparse')
# parser.add_argument('-姓名','--name',default='覃皓')
# parser.add_argument('-年龄','--year',default='22')
# args = parser.parse_args()
#
# print(args)
#
# name = args.name
#
# year = args.year
#
# print('hello {} {}'.format(name,year))


# import sys
# print(sys.path)
# print(dir(sys))


# class demo:
#     def __init__(self):
#         self.name = 'dd'
#
#     def __getattr__(self, attrname):
#         if attrname == "age":
#             return 40
#         else:
#             raise (attrname)
#
#
# x = demo()
# print(x.age)
# print(x.name)

# import numpy as np
# from matplotlib import patches
# import matplotlib.pyplot as plt
# from matplotlib.collections import PatchCollection
#
# # 绘制一个椭圆需要制定椭圆的中心，椭圆的长和高
# xcenter, ycenter = 1, 1
# width, height = 0.8, 0.5
# angle = -30  # 椭圆的旋转角度
#
# fig = plt.figure()
# ax = fig.add_subplot(211, aspect='auto')
# ax.set_xbound(-1, 4)
# ax.set_ybound(-1, 4)
#
# e1 = patches.Ellipse((0, 0), width, height,
#                      angle=angle, linewidth=2, fill=False, zorder=2)
#
# e2 = patches.Arc((2, 2), width=3, height=2,
#                  angle=angle, linewidth=2, fill=False, zorder=2)
#
# patches = []
# patches.append(e1)
# patches.append(e2)
# collection = PatchCollection(patches)
# ax.add_collection(collection)
#
# plt.show()
# import time
#
# t = (2009, 2, 17, 17, 3, 38, 1, 48, 0)
# t = time.mktime(t)
# print(time.strftime("%b %d %Y %H:%M:%S", time.gmtime(t)))
# import torch
# print(torch.__version__)
# print(torch.cuda.is_available())
# import functools
#
# def add(a,b):
#     print(a+b)
#
# add=functools.partial(add,1)
# add(2)
# from collections import deque
#
# class Solution:
#     def highestPeak(self, isWater):
#         m, n = len(isWater), len(isWater[0])
#         ans = [[0] * n for _ in range(m)]
#         d = deque()
#         for i in range(m):
#             for j in range(n):
#                 if isWater[i][j]:
#                     d.append((i, j))
#                 ans[i][j] = 0 if isWater[i][j] else -1
#         # import ipdb;ipdb.set_trace()
#         dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
#         h = 1
#         while d:
#             size = len(d)
#             for _ in range(size):
#                 x, y = d.popleft()
#                 for di in dirs:
#                     nx, ny = x + di[0], y + di[1]
#                     if 0 <= nx < m and 0 <= ny < n and ans[nx][ny] == -1:
#                         ans[nx][ny] = h
#                         d.append((nx, ny))
#             h += 1
#         return ans
#
#
# a=[[0,0,1],[1,0,0],[0,0,0]]
# b=Solution()
#
# print(b.highestPeak(a))
# import torch as t
# t1=[t.tensor([1,2]),t.tensor([3,4])]
# t2=t.cat(t1,dim=1)
# print(t2)
# t1=t.randn((1,3,2,2))
# print(t1)
# t1_mean=t.mean(t1,dim=1,keepdim=True)
#
# print(t1_mean)

# import numpy as np
# from PIL import Image
#

#
# def pca(data,k):
#     """
#     :param data:  图像数据
#     :param k: 保留前k个主成分
#     :return:
#     """
#     import ipdb;ipdb.set_trace()
#     n_samples, n_features = data.shape
#     # 求均值
#     mean = np.array([np.mean(data[:,i]) for i in range(n_features)])
#     # 去中心化
#     normal_data = data - mean
#     # 得到协方差矩阵
#     matrix_ = np.dot(np.transpose(normal_data),normal_data)
#     # 有时会出现复数特征值，导致无法继续计算，这里用了不同的图像，有时候会出现复数特征，但是经过
#     # sklearn中的pca处理同样图像可以得到结果，如果你知道，请留言告诉我。
#     # 我能知道的是协方差矩阵肯定是实对称矩阵的
#     eig_val,eig_vec = np.linalg.eig(matrix_)
#    # print(matrix_.shape)
#    #  print(eig_val)
#     # 第一种求前k个向量
#    #  eig_pairs = [(np.abs(eig_val[i]),eig_vec[:,i]) for i in range(n_features)]
#    #  eig_pairs.sort(reverse=True)
#    #  feature = np.array([ele[1] for ele in eig_pairs[:k]])
#    #  new_data = np.dot(normal_data,np.transpose(feature))
#     # 第二种求前k个向量
#     eigIndex = np.argsort(eig_val)
#     eigVecIndex = eigIndex[:-(k+1):-1]
#     feature = eig_vec[:,eigVecIndex]
#     new_data = np.dot(normal_data,feature)
#     # 将降维后的数据映射回原空间
#     rec_data = np.dot(new_data,np.transpose(feature))+ mean
#     # print(rec_data)
#     # 压缩后的数据也需要乘100还原成RGB值的范围
#     newImage = Image.fromarray(rec_data*100)
#     newImage.show()
#     return rec_data
#
#
# pic_root='./1.jpg'
# pic_data=loadImage(pic_root)
# rec_data=pca(pic_data,k=64)

# import numpy as np
# from sklearn.decomposition import PCA
#
# import numpy as np
# from PIL import Image
#
#
# if __name__ == '__main__':
#     data = loadImage("1.jpg")
#     pca = PCA(n_components=64).fit(data)
#     # 降维
#     x_new = pca.transform(data)
#     # 还原降维后的数据到原空间
#     recdata = pca.inverse_transform(x_new)
#     # 计算误差
#     # error(data, recdata)
#     # 还原降维后的数据
#     newImg = Image.fromarray(recdata*100)
#     newImg.show()

# import cv2
# import numpy as np
# from PIL import Image

# def loadImage(path):
#     img = Image.open(path)
#     # 将图像转换成灰度图
#     img = img.convert("L")
#     # 图像的大小在size中是（宽，高）
#     # 所以width取size的第一个值，height取第二个
#     width = img.size[0]
#     height = img.size[1]
#     data = img.getdata()
#     # 为了避免溢出，这里对数据进行一个缩放，缩小100倍
#     data = np.array(data).reshape(height, width) / 100
#     # 查看原图的话，需要还原数据
#     new_im = Image.fromarray(data * 100)
#     # import ipdb;ipdb.set_trace()
#     # new_im.show()
#     return data
#
#
# if __name__ == '__main__':
#     img = cv2.imread('1.jpg')
#
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     image_filter=cv2.medianBlur(gray,3)
#
#     image_canny=cv2.Canny(image_filter,20,100)
#     # image_canny=image_canny/255
#     # import ipdb;ipdb.set_trace()
#     mask_canny=cv2.bitwise_and(img.transpose((2,0,1))[2],image_canny)
#     cv2.imshow('mask',mask_canny)
#     cv2.imshow('gray', gray)
#     cv2.imshow('filter',image_filter)
#
#     cv2.imshow('canny',image_canny)
#     cv2.waitKey(0)

# import cv2
# import numpy as np
#
# # 载入灰度原图，并且归一化
# img_original=cv2.imread('F:\\VOCdevkit\\VOCdevkit\\VOC2007\\JPEGImages\\000002.jpg',0)/255
# #分别求X,Y方向的梯度
# grad_X=cv2.Sobel(img_original,-1,1,0)
# grad_Y=cv2.Sobel(img_original,-1,0,1)
# #求梯度图像
# grad=cv2.addWeighted(grad_X,0.5,grad_Y,0.5,0)
# cv2.imshow('gradient',grad)
# cv2.waitKey()
# cv2.destroyAllWindows()
# # img_original = cv2.imread('F:\\VOCdevkit\\VOCdevkit\\VOC2007\\JPEGImages\\000002.jpg', 0)
# # # 求X方向梯度，并且输出图像一个为CV_8U,一个为CV_64F
# # img_gradient_X_8U = cv2.Sobel(img_original, -1, 1, 0)
# # img_gradient_X_64F = cv2.Sobel(img_original, cv2.CV_64F, 1, 0)
# # # 将图像深度改为CV_8U
# # img_gradient_X_64Fto8U = cv2.convertScaleAbs(img_gradient_X_64F)
# # # 图像显示
# # cv2.imshow('X_gradient_8U', img_gradient_X_8U)
# # cv2.imshow('X_gradient_64Fto8U', img_gradient_X_64Fto8U)
# # cv2.waitKey()
# # cv2.destroyAllWindows()
import cv2
o = cv2.imread("F:\\VOCdevkit\\VOCdevkit\\VOC2007\\JPEGImages\\000002.jpg", 0)
# o = cv2.imread("F:\\VOCdevkit\\VOCdevkit\\VOC2007\\JPEGImages\\000002.jpg", cv2.IMREAD_GRAYSCALE)
r1 = cv2.Canny(o, 10, 100)
r2 = cv2.Canny(o, 32, 128)
cv2.imshow("original", o)
cv2.imshow("result1", r1)
cv2.imshow("result2", r2)
cv2.waitKey()
cv2.destroyAllWindows()
