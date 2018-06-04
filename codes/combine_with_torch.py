from __future__ import print_function
from skimage import io
from skimage.util import view_as_windows
import os
import numpy as np
from tqdm import tqdm
import argparse
import glob

if __name__ == "__main__":

    # This is for obtaining square images
    # TODO: For non-square images implement, rewrite print statement in run 
    #python create_patch.py --input_path='/media/balamurali/Seagate Backup Plus Drive/Mitosis_detection/2012_aperio_scan_image/whole_images' --output_path='/media/balamurali/Seagate Backup Plus Drive/Mitosis_detection/2012_aperio_scan_image/sampled_images' --img_ext='bmp' --stride=32 --img_size=256 --img_size=512
    #python create_patch.py --input_path='/media/htic/Seagate Backup Plus Drive/Mitosis_detection/2012_aperio_scan_image/mitosis_evaluation_mask' --output_path='/media/htic/Seagate Backup Plus Drive/Mitosis_detection/2012_aperio_scan_image/dataset/test_data/' --img_ext='bmp' --stride=512 --img_size=512


    parser = argparse.ArgumentParser('Sample images giving image size and stride -- Please give absolute path')
    parser.add_argument(
        '--input_path',
        required = True,
        type = str,
        help = 'provide path which contains images' )

    parser.add_argument(
        '--output_path',
        required = True,
        type = str,
        help = 'provide path to save the images')

    parser.add_argument(
        '--img_ext',
        required = True,
        type = str,
        help = 'provide extension of the image')

    parser.add_argument(
        '--stride',
        required = True,
        type = int,
        action = 'append',
        help = 'stride length'
    )

    parser.add_argument(
        '--img_size',
        required = True,
        help = 'give the image dimensions',
        action = 'append',
        type = int
    )

    opt = parser.parse_args()    
    img_folder = opt.input_path
    # img_folder = '/media/balamurali/NewVolume2/Deep_network/mitosis'
    out_folder = opt.output_path
    # out_folder = '/media/balamurali/NewVolume2/Deep_network/mitosis/output'
    img_ext    = opt.img_ext
    img_size   = opt.img_size
    img_stride = opt.stride
    
    stride_list = []
    window_shape_list = []
    for each in img_size:
        window_shape_list.append((each,each))
        
    for each in img_stride:
        stride_list.append(each)


    print ("Settings: \nInput Folder:{}\nOutput Folder:{}\nImg ext:{}\nImg size:{}\n".format(
        img_folder,out_folder,img_ext,window_shape_list))


    imgs_path  = glob.glob(os.path.join(img_folder,'*.' + img_ext))    
    for window_shape in tqdm(window_shape_list):
        print ("Window size : {}\n".format(window_shape))
        
        for stride in tqdm(stride_list):
            print ("Stride size : {}\n".format(stride))
                
            out_folder_temp = os.path.join(out_folder,'size_' + str(window_shape[0]) + '_' + 'stride_' +str(stride))    
        
            if not os.path.exists(out_folder_temp):
                print ("Directory {} not found,creating one".format(out_folder_temp))
                os.mkdir(out_folder_temp)
        
            print ("Sampling images\n")
            for img_path in tqdm(imgs_path):
                # Read image
                img = io.imread(img_path)
                print (len(img.shape))
                if len(img.shape) == 3:
                    r_channel = img[:,:,0]
                    g_channel = img[:,:,1]
                    b_channel = img[:,:,2]

                    r_sample = view_as_windows(r_channel,window_shape,step=stride)
                    g_sample = view_as_windows(g_channel,window_shape,step=stride)
                    b_sample = view_as_windows(b_channel,window_shape,step=stride)
                else:
                    r_sample = view_as_windows(img,window_shape,step=stride)

                no_of_rows = r_sample.shape[0]
                no_of_cols = r_sample.shape[1]

                for row in tqdm(range(no_of_rows)):
                    for col in tqdm(range(no_of_cols)):
                        if len(img.shape) == 3:
                            sample_r = r_sample[row,col]
                            sample_g = g_sample[row,col]
                            sample_b = b_sample[row,col]
                            img_sample = np.dstack((sample_r,sample_g,sample_b))
                        else:
                            img_sample = r_sample[row,col]  
                        # print (img_sample.shape)
                        # img_sample_path = (os.path.join(out_folder_temp,os.path.basename(img_path)[:-4] + '_{}_{}.jpg')).format(row,col)
                        # io.imsave(img_sample_path,img_sample)
                        # break
                    # break
                # break
            # break