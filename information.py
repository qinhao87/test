import numpy as np
from data.dataset import TestDataset

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
def inf(ii,pictrue_name,gt_labels,gt_difficult,gt_bbox,pred_labels,pred_bboxes,pred_score):

    pictrue_name=pictrue_name

    gt_labels=gt_labels.tolist()
    gt_labels=[int(gt_label) for gt_label in gt_labels]
    gt_labels=[VOC_BBOX_LABEL_NAMES[gt_label] for gt_label in gt_labels]

    gt_difficult=gt_difficult

    gt_bbox=gt_bbox

    pred_labels=pred_labels.tolist()
    pred_labels=[int(pred_label) for pred_label in pred_labels]
    pred_labels=[VOC_BBOX_LABEL_NAMES[pred_label] for pred_label in pred_labels]

    pred_bbox=pred_bboxes

    pred_score=pred_score

    filename='information.txt'
    with open(filename,'a',encoding='utf-8') as f:
        f.write('图片名称:{}\ngt_information:\n'.format(pictrue_name))
        f.write('gt_label:{},gt_difficult:{},gt_bbox:{}\n'.format(gt_labels,gt_difficult,gt_bbox))
        f.close()
        f.write('pred_information:\npred_label:{},pred_bbox:{},pred_score:{}\n'.format(pred_labels,pred_bbox,pred_score))
        f.close()
