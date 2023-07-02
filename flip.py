import sys
import cv2
import os
from PIL import Image
from PIL import ImageDraw

os.getcwd()
im_path=os.path.join(os.getcwd(),"005.png")
print(im_path)
img1=Image.open(im_path)
img1.show()
cv2.waitKey(5)
out1=img1.transpose(Image.FLIP_LEFT_RIGHT)
out1.show()