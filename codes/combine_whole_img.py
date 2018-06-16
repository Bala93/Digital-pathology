'''
Code is used to convert separate images to whole image - the result is from pix2pix
'''

import glob
import argparse
import os
from tqdm import tqdm
import cv2
import numpy as np
from tqdm import tqdm

if __name__ == "__main__":

    #glob.glob('')
    
    no_test_images = 15
    sample_out_path = '/media/htic/NewVolume1/murali/pytorch-CycleGAN-and-pix2pix/results/mitotic_pix2pix/test_5/images'
    sample_ext  = 'png'
    save_dir = '/media/htic/NewVolume1/murali/mitosis/mitotic_segment/results'
    save_ext = 'bmp'
    gt_path = '/media/htic/NewVolume1/murali/mitosis/mitotic_count/test_masks'
    gt_ext = 'bmp'
    
    stride = 183
    w = 256
    h = 256

    no_of_rows = 10
    no_of_cols = 10

    for test_img_no in tqdm(range(no_test_images)):
        whole_mask = np.zeros((2084,2084,3),np.uint8)

        for row in tqdm(range(no_of_rows)):
            for col in tqdm(range(no_of_cols)):
                mask_name      = '{}_{}_{}_real_A.{}'.format(test_img_no + 1,row,col,sample_ext)

                mask_path_file = os.path.join(sample_out_path,mask_name)

                # print (mask_path_file)
                # Verify the binary map
                mask_img = cv2.imread(mask_path_file)
                # _,th   = cv2.threshold(mask_img,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                ####
                row_start = row * stride
                row_end   = row_start + h
                col_start = col * stride 
                col_end   = col_start + w

                whole_mask[row_start:row_end,col_start:col_end,:] = cv2.addWeighted(whole_mask[row_start:row_end,col_start:col_end,:],0.5,mask_img,0.5,0)
                # whole_mask[row_start:row_end,col_start:col_end,:] /= 2

        # kernel = np.ones((5,5),np.uint8)

        # whole_mask = cv2.dilate(whole_mask,kernel,iterations=5)

        # save_file_name = '{}.{}'.format(test_img_no + 1,save_ext)
        # save_path = os.path.join(save_dir,save_file_name)
        # gt_file_name = '{}.{}'.format(test_img_no + 1,gt_ext)
        # gt_mask = cv2.imread(os.path.join(gt_path,gt_file_name),0)


        # _,whole_mask_bin = cv2.threshold(whole_mask,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # _,gt_mask_bin = cv2.threshold(gt_mask,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # print whole_mask_bin.shape
        # print gt_mask_bin.shape

        # overlap_mask = np.bitwise_and(whole_mask_bin,gt_mask_bin) #cv2.bitwise_and(whole_mask_bin,gt_mask_bin,)

        cv2.namedWindow('pred',cv2.WINDOW_NORMAL)
        # cv2.namedWindow('gt',cv2.WINDOW_NORMAL)
        # cv2.namedWindow('overlap',cv2.WINDOW_NORMAL)

        cv2.imshow('pred',whole_mask)
        # cv2.imshow('gt',gt_mask_bin)
        # cv2.imshow('overlap',overlap_mask)

        cv2.waitKey(0)
        cv2.destroyAllWindows()


        # print (save_path)
        cv2.imwrite('/home/htic/Desktop/out.jpg',whole_mask)
        break