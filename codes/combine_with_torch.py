from __future__ import print_function
from skimage import io
from skimage.util import view_as_windows
import os
import numpy as np
from tqdm import tqdm
import argparse
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
from PIL import Image


if __name__ == "__main__":

    # This is for obtaining square images
    # TODO: For non-square images implement, rewrite print statement in run 
    #python create_patch.py --input_path='/media/balamurali/Seagate Backup Plus Drive/Mitosis_detection/2012_aperio_scan_image/whole_images' --output_path='/media/balamurali/Seagate Backup Plus Drive/Mitosis_detection/2012_aperio_scan_image/sampled_images' --img_ext='bmp' --stride=32 --img_size=256 --img_size=512
    #python create_patch.py --input_path='/media/htic/NewVolume1/murali/mitosis/dataset/eval/' --img_ext='png' --stride=512 --img_size=512


    parser = argparse.ArgumentParser('Sample images giving image size and stride -- Please give absolute path')
    parser.add_argument(
        '--input_path',
        required = True,
        type = str,
        help = 'provide path which contains images' )

    # parser.add_argument(
    #     '--output_path',
    #     # required = True,
    #     type = str,
    #     help = 'provide path to save the images')

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
    # out_folder = opt.output_path
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


    model_path  = '/media/htic/NewVolume1/murali/mitosis/weight/whole_slide.pt'
    model_ft = models.resnet101()
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs,2)
    model_ft.load_state_dict(torch.load(model_path))
    model_ft.cuda()

    data_transforms = {
    'train': transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    }


    # print ("Settings: \nInput Folder:{}\nOutput Folder:{}\nImg ext:{}\nImg size:{}\n".format(
        # img_folder,out_folder,img_ext,window_shape_list))


    imgs_path  = glob.glob(os.path.join(img_folder,'*.' + img_ext))    

    for window_shape in tqdm(window_shape_list):
        print ("Window size : {}\n".format(window_shape))
        
        for stride in tqdm(stride_list):
            print ("Stride size : {}\n".format(stride))
                
            # out_folder_temp = os.path.join(out_folder,'size_' + str(window_shape[0]) + '_' + 'stride_' +str(stride))    
        
            # if not os.path.exists(out_folder_temp):
            #     print ("Directory {} not found,creating one".format(out_folder_temp))
            #     os.mkdir(out_folder_temp)
        
            print ("Sampling images\n")
            for img_path in tqdm(imgs_path):
                # Read image
                img = io.imread(img_path)
                # print (len(img.shape))
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

                        # img_sample = np.swapaxes(img_sample,0,1)
                        # img_sample = np.swapaxes(img_sample,0,2)
                        # img_sample = torch.FloatTensor(img_sample)
                        # img_sample = img_sample.unsqueeze(0)
                        print (img_sample.shape)
                        img_sample = Image.fromarray(img_sample)
                        img_sample = data_transforms['val'](img_sample)
                        print (img_sample.shape )

                        break
                    break
                break
            break
        break
                            # inputs = Variable(img_s.cuda())

                        # outputs = model_ft(inputs)
                        # _, preds = torch.max(outputs.data, 1)

                        # y_true.append(labels.data.cpu().numpy()) 
                        # y_pred.append(preds.cpu().numpy())

                        # print (y_pred[0][0],y_true[0][0])



                        # print (img_sample.shape)
                        # img_sample_path = (os.path.join(out_folder_temp,os.path.basename(img_path)[:-4] + '_{}_{}.jpg')).format(row,col)
                        # io.imsave(img_sample_path,img_sample)
                        # break
                    # break
                # break
            # break

# use_gpu = torch.cuda.is_available()
# y_true = []
# y_pred = []




# plt.imshow(inputs.cpu())
# break