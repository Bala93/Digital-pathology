from skimage import io
from skimage.transform import rescale
import glob
import os
from tqdm import tqdm
import argparse


## The mask and image are assumed to be of same dimensions.
'''
python resize_save.py --input_img_path --input_mask_path --output_img_path --output_mask_path --img_ext --scale
'''

parser = argparse.ArgumentParser(description='Resize images in a folder and move to new folder')

parser.add_argument('--input_img_path',required = True, type = str, help = 'provide input_path file')
parser.add_argument('--input_mask_path',required = True, type = str, help = 'provide input_path file')

parser.add_argument('--output_img_path',required = True, type = str,help = 'provide output_path file')
parser.add_argument('--output_mask_path',required = True, type = str,help = 'provide output_path file')

parser.add_argument('--img_ext',required = True, type = str,help='provide image extension')

parser.add_argument('--scale',required = True , type = int,help = 'rescale factor')

opt = parser.parse_args()

src_img_path = opt.input_img_path
dst_img_path = opt.output_img_path
src_mask_path = opt.input_mask_path
dst_mask_path = opt.output_mask_path

img_ext     = opt.img_ext
#mask_ext    = opt.mask_ext 
scale       = opt.scale

src_imgs = glob.glob(src_img_path + '/*' + img_ext)

for img_path in tqdm(src_imgs):
    file_name = os.path.basename(img_path)

    src_img  = img_path
    src_mask = os.path.join(src_mask_path,file_name)
    dst_img  = os.path.join(dst_img_path,file_name)
    dst_mask = os.path.join(dst_mask_path,file_name)

    img = io.imread(src_img)
    mask = io.imread(src_mask)

    img_rescale  = rescale(img,scale,order=3)
    mask_rescale = rescale(mask,scale,order=3)


    io.imsave(dst_img,img_rescale)
    io.imsave(dst_mask,mask_rescale)

    # break
