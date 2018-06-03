
from __future__ import print_function
import glob
import os
import random
import shutil as sh
from tqdm import tqdm

if __name__ == "__main__":
    
    '''
        This tool if provided with two folders with original image and the mask, then it gets it ready 
        for pix2pix type dataset. 
    '''

    # This has to received from the user. 
    img_ext     = 'jpg'
    mask_ext    = 'png'
    img_path    = '/media/balamurali/New Volume2/IIT-HTIC/GE_Project/Pix2Pix/Pix2Pix-Dataset'
    mask_path   = '/media/balamurali/New Volume2/IIT-HTIC/GE_Project/Pix2Pix/Pix2Pix-DatasetBinaryMask'
    out_path    = '/media/balamurali/New Volume2/IIT-HTIC/GE_Project/Pix2Pix'
    split_ratio = 0.8 # Train and test split
    #######################

    # A - Image, B - Mask
    # Setting the input and output -- Before doing the dataset arrangement.
    in_img_path     = os.path.join(img_path , '*.' + img_ext)
    in_mask_path    = os.path.join(mask_path , '*.' + mask_ext)
    out_img_path    = os.path.join(out_path,'A')
    out_mask_path   = os.path.join(out_path,'B')
    ######################

    # Create image and mask directory
    if not os.path.exists(out_img_path):
        os.mkdir(out_img_path)
    if not os.path.exists(out_mask_path):
        os.mkdir(out_mask_path)
    #############################


    # Create train and test folder inside A (image) and B (mask)
    train_A_path =  os.path.join(out_img_path,'train')
    train_B_path = os.path.join(out_mask_path,'train')
    test_A_path  = os.path.join(out_img_path,'test')
    test_B_path  = os.path.join(out_mask_path,'test')

    if not os.path.exists(train_A_path):
        os.mkdir(train_A_path)
    if not os.path.exists(train_B_path):
        os.mkdir(train_B_path)

    if not os.path.exists(test_A_path):
        os.mkdir(test_A_path)
    if not os.path.exists(test_B_path):
        os.mkdir(test_B_path)
    #############################

    # Create train and val
    imgs = glob.glob(in_mask_path)
    print ("Number of images: {}".format(len(imgs)))
    random.shuffle(imgs)
    train_len = int(len(imgs) * split_ratio)

    print ("Train images/masks are getting ready")
    for src_path in tqdm(imgs[:train_len]):
        src_path_split = src_path.split('/')
        img_name = src_path_split[-1][:-4]

        src_mask  = src_path
        src_img = os.path.join(img_path,img_name + '.' + img_ext )
        dst_img  = os.path.join(train_A_path,img_name + '.' + img_ext)
        dst_mask = os.path.join(train_B_path,img_name + '.' + img_ext)
        # print (src_img)
        # print (src_mask)
        # print (dst_img)
        # print (dst_mask)
        sh.copy(src_img,dst_img)
        sh.copy(src_mask,dst_mask)

    print ("Test images/masks are getting ready")
    for src_path in tqdm(imgs[train_len:]):
        src_path_split = src_path.split('/')
        img_name = src_path_split[-1][:-4]

        src_mask  = src_path
        src_img = os.path.join(img_path,img_name + '.' + img_ext )
        dst_img  = os.path.join(test_A_path,img_name + '.' + img_ext)
        dst_mask = os.path.join(test_B_path,img_name + '.' + img_ext)
        # print (src_img)
        # print (src_mask)
        # print (dst_img)
        # print (dst_mask)
        sh.copy(src_img,dst_img)
        sh.copy(src_mask,dst_mask)
