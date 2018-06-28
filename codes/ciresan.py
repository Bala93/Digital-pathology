# /media/htic/NewVolume1/murali/mitosis/mitotic_count/whole_images
# /media/htic/NewVolume1/murali/mitosis/mitotic_count/whole_mask


from __future__ import print_function
from skimage import io
from skimage import util 
from skimage import morphology
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

    # python ciresan.py --input_path='/media/htic/NewVolume1/murali/mitosis/mitotic_count/whole_images_normalization_Macenko' --output_path='/media/htic/NewVolume1/murali/mitosis/mitotic_count/ciresan_dataset' --img_ext='bmp' --stride=1 --img_size=101 --input_type='image' --pad=50 --pad_mode='symmetric'

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

    parser.add_argument(
        '--input_type',
        required = True,
        help = 'Give mask or image',
        type = str
    )

    parser.add_argument(
        '--pad',
        required = False,
        default = 0,
        help = 'Pad given data(optional)',
        type = int
    )

    parser.add_argument(
        '--pad_mode',
        required = False,
        default = 'constant',
        help = 'Pad mode(optional)',
        type = str
    )


    opt = parser.parse_args()    
    img_folder = opt.input_path
    # img_folder = '/media/balamurali/NewVolume2/Deep_network/mitosis'
    out_folder = opt.output_path
    # out_folder = '/media/balamurali/NewVolume2/Deep_network/mitosis/output'
    img_ext    = opt.img_ext
    img_size   = opt.img_size
    img_stride = opt.stride
    input_type = opt.input_type
    pad_width  = opt.pad
    pad_mode   = opt.pad_mode

    mask_centroid = '/media/htic/NewVolume1/murali/mitosis/mitotic_count/whole_mask_centroid'

    #print(pad,pad_mode) 
    
    stride_list = []
    window_shape_list = []
    for each in img_size:
        window_shape_list.append((each,each))
        
    for each in img_stride:
        stride_list.append(each)


    print ("Settings: \n Input Folder:{}\n Output Folder:{}\n Img ext:{}\n Img size:{}\n Stride:{}\n Padding:{}\n Padmode:{}".format(
        img_folder,out_folder,img_ext,window_shape_list,stride_list,pad_width,pad_mode))


    imgs_path  = glob.glob(os.path.join(img_folder,'*.' + img_ext))    
    for window_shape in tqdm(window_shape_list):
        # print ("Window size : {}\n".format(window_shape))
        
        for stride in tqdm(stride_list):
            # print ("Stride size : {}\n".format(stride))
                
            out_folder_temp = os.path.join(out_folder,input_type + '_size_' + str(window_shape[0]) + '_' + 'stride_' +str(stride))    
        
            if not os.path.exists(out_folder_temp):
                print ("Directory {} not found,creating one".format(out_folder_temp))
                os.mkdir(out_folder_temp)

            out_folder_temp_mitosis = os.path.join(out_folder_temp,'mitosis')
            out_folder_temp_non_mitosis = os.path.join(out_folder_temp,'non_mitosis')

            if not os.path.exists(out_folder_temp_mitosis):
                os.mkdir(out_folder_temp_mitosis)

            if not os.path.exists(out_folder_temp_non_mitosis):
                os.mkdir(out_folder_temp_non_mitosis)
                
            print ("Sampling images\n")
            for img_path in tqdm(imgs_path):
                
                img_name = os.path.basename(img_path)
                mask_centroid_path = os.path.join(mask_centroid,img_name)
                centroid_mask = io.imread(mask_centroid_path,as_grey=True)
                centroid_mask = util.pad(centroid_mask,((pad_width,pad_width),(pad_width,pad_width)),pad_mode)
                #centroid_mask = morphology.binary_dilation(centroid_mask,morphology.square(3))

                #io.imshow(centroid_mask)
                #io.show()
                centroid_mask /= 255
                print ("No. of cells",np.sum(centroid_mask))
                print (np.where(centroid_mask==1))


                # Read image
                img = io.imread(img_path)
                

                #Pad image
                if(pad_width):
                    img = util.pad(img,((pad_width,pad_width),(pad_width,pad_width),(0,0)),pad_mode)
        
               
                r_channel = img[:,:,0]
                g_channel = img[:,:,1]
                b_channel = img[:,:,2]

                r_sample = util.view_as_windows(r_channel,window_shape,step=stride)
                g_sample = util.view_as_windows(g_channel,window_shape,step=stride)
                b_sample = util.view_as_windows(b_channel,window_shape,step=stride)
               
                m_sample = util.view_as_windows(centroid_mask,window_shape,step=stride)

                no_of_rows = r_sample.shape[0]
                no_of_cols = r_sample.shape[1]
                print ("{}x{} images to be created".format(no_of_rows,no_of_cols))

                for row in tqdm(range(no_of_rows)):
                    for col in tqdm(range(no_of_cols)):
                        
                        sample_r = r_sample[row,col]
                        sample_g = g_sample[row,col]
                        sample_b = b_sample[row,col]
                        img_sample = np.dstack((sample_r,sample_g,sample_b))
                        mask_sample = m_sample[row,col]
                        roi = mask_sample[40:60,40:60]                
                        #print (roi.shape)
                        
                        if np.sum(roi) > 0:
                            out_folder_save  = out_folder_temp_mitosis
                            #print ("Added file in mitosis folder")
                        else:
                            out_folder_save  = out_folder_temp_non_mitosis

                        img_sample_path = (os.path.join(out_folder_save,os.path.basename(img_path)[:-4] + '_{}_{}.jpg')).format(row,col)
                        io.imsave(img_sample_path,img_sample)

                        #break
                    #break
                break
            break
print('finish')
