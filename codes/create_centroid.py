import cv2
import numpy as np
import glob


if __name__ == "__main__":


    mask_path = ''
    mask_ext  = ''
    mask_path_ext = os.path.join(mask_path , '*.' + mask_ext)
    mask_images_path = glob.glob(mask_path_ext)

    for each_mask_path in glob.glob():
        img_mask = cv2.imread(each_mask_path)
        img_mask_gray = cv2.cvtColor(img_mask,cv2.COLOR_BGR2GRAY)
        _,im_bw = cv2.threshold(img_mask_gray, 200, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        kernel = np.ones([5,5],np.uint8)
        dilation = cv2.dilate(im_bw,kernel,iterations=3)
        cnts = cv2.findContours(dilation, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]