from skimage import io
from skimage.transform import rescale
import glob
import os
from tqdm import tqdm

src_img_path = '/media/htic/NewVolume1/murali/mitosis/mitotic_count/image_size_256_stride_32_updated'
dst_img_path = '/media/htic/NewVolume1/murali/mitosis/mitotic_count/image_resize_512_stride_32_updated'
src_mask_path = '/media/htic/NewVolume1/murali/mitosis/mitotic_count/mask_size_256_stride_32_updated'
dst_mask_path = '/media/htic/NewVolume1/murali/mitosis/mitotic_count/mask_resize_512_stride_32_updated'

img_ext = 'jpg'

src_imgs = glob.glob(src_img_path + '/*' + img_ext)

for img_path in tqdm(src_imgs):
    file_name = os.path.basename(img_path)

    src_img  = img_path
    src_mask = os.path.join(src_mask_path,file_name)
    dst_img  = os.path.join(dst_img_path,file_name)
    dst_mask = os.path.join(dst_mask_path,file_name)


    img = io.imread(src_img)
    mask = io.imread(src_mask)

    img_rescale  = rescale(img,2,order=3)
    mask_rescale = rescale(mask,2,order=3)


    io.imsave(dst_img,img_rescale)
    io.imsave(dst_mask,mask_rescale)

    # break
