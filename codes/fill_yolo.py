import cv2
import glob
from tqdm import tqdm
import json
import os
import numpy as np

with open('/media/htic/NewVolume1/murali/mitosis/mitotic_count/ground_truth_boxes.json') as f: 
    json_dict = json.load(f)

if __name__=="__main__":

    img_dir = '/media/htic/NewVolume1/murali/mitosis/mitotic_count/test_images/*.bmp'
    out_path = '/media/htic/NewVolume1/murali/mitosis/mitotic_count/yolo_input_gt'
    img_path = glob.glob(img_dir)
    eps = 75 

    for img_name in img_path:
        fname = os.path.basename(img_name)
        img = cv2.imread(img_name)
        mask = 255 * np.ones(img.shape,dtype=np.uint8)
        # for coord in json_dict[fname]['boxes']:
        for coord in json_dict[fname]:
            xmin,ymin,xmax,ymax = coord
            mask[ymin-eps:ymax+eps,xmin-eps:xmax+eps,:] = 0
        
        upd_img = cv2.add(img,mask)
        cv2.imwrite(os.path.join(out_path,fname),upd_img)

        # cv2.namedWindow('win',cv2.WINDOW_NORMAL)
        # cv2.imshow('win',upd_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
            # break
        # break
        # cv2.add()
