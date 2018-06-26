import glob
from skimage import io
import numpy as np
import os
from tqdm import tqdm
import cv2

mask_path = '/media/htic/NewVolume1/murali/mitosis/mitotic_count/mask_size_512_stride_32_updated/*.jpg'
img_path = '/media/htic/NewVolume1/murali/mitosis/mitotic_count/img_size_512_stride_32_updated/'

masks =glob.glob(mask_path)

for i in tqdm(masks):
    j = i.replace('mask_size_512_stride_32_updated','image_size_512_stride_32_updated')
    # print i,j
    img = cv2.imread(i,0)
    _,img = cv2.threshold(img,200,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    img = img/255
    #print(img.shape)
    #print (np.sum(img))
    if np.sum(img) < 150:
        #print(i)
        os.remove(i)
        os.remove(j)
    # break
