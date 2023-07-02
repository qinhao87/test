from pycocotools.coco import COCO
import cv2
import numpy as np

# 加载COCO数据集
dataDir = 'path/to/coco/dataset'
dataType = 'train2017'
annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)
coco = COCO(annFile)

# 获取图像信息和标注信息
img_id = 1000  # 示例图像ID
img_info = coco.loadImgs(ids=[img_id])[0]
ann_ids = coco.getAnnIds(imgIds=[img_id])
ann_info = coco.loadAnns(ids=ann_ids)

# 加载图像并将标注信息可视化
img_path = '{}/images/{}/{}'.format(dataDir, dataType, img_info['file_name'])
img = cv2.imread(img_path)

for ann in ann_info:
    # 为每个标注区域创建掩膜
    mask = coco.annToMask(ann)
    # 将掩膜转换为二进制图像
    mask = np.array(mask, dtype=np.uint8)
    mask *= 255
    # 在图像上绘制分割标注部分
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, contours, -1, (0, 0, 255), 2)

# 显示图像和分割标注部分
cv2.imshow('Image', img)
cv2.waitKey(0)
