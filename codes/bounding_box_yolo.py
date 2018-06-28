import xml.etree.ElementTree as et
import glob
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import imutils
import shutil
from tqdm import tqdm


# def display_img(img):
#     cv2.imshow('win',img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#     return

# def write_img(img,img_path):
#     cv2.imwrite(img_path,img)
#     return

# def write_xml(root,xmin,ymin,xmax,ymax,width,height):
    
#     xmin_tag = et.Element('xmin')
#     ymin_tag = et.Element('ymin')
#     xmax_tag = et.Element('xmax')
#     ymax_tag = et.Element('ymax')
#     bndbox = root.find('object').find('bndbox')
#     bndbox.insert(1,xmin_tag)
#     bndbox.insert(1,ymin_tag)
#     bndbox.insert(1,xmax_tag)
#     bndbox.insert(1,ymax_tag)
    
    # if(xmin>width):
    #     xmin = width 
    # if(ymin>height):
    #     ymin = height 
    # if(xmax>width):
    #     xmax = width 
    # if(ymax>height):
    #     ymax = height 
    
    # if(xmin<0):
    #     xmin = 0
    # if(ymin<0):
    #     ymin = 0
    # if(xmax<0):
    #     xmax = 0
    # if(ymax<0):
    #     ymax = 0
    
#     bndbox.find('xmin').text = str(xmin)
#     bndbox.find('ymin').text = str(ymin)
#     bndbox.find('xmax').text = str(xmax)
#     bndbox.find('ymax').text = str(ymax)        

#     return 


if __name__ == "__main__":
    
    '''
    This works when there is no overlap between mask regions. TODO: See if the can be extended to overlapping regions also.
    Example : 
    python bounding_box_create.py --mask_path='/media/htic/NewVolume1/murali/mitosis/mitotic_count/mask_size_512_stride_128_updated' --mask_ext='jpg' --xml_path='/media/htic/NewVolume1/murali/mitosis/512_128_xml'
    TODO: Add the height and width using the mask size
    '''

    # Receive inputs from user.
    parser = argparse.ArgumentParser('Convert mask images to xml files')
    parser.add_argument(
        '--mask_path',
        required = True,
        type = str,
        help = 'Path which contains mask files'
    )

    parser.add_argument(
        '--mask_ext',
        required = True,
        type = str,
        help = 'Mask file extension'
    )

    parser.add_argument(
        '--out_path',
        required = True,
        type = str,
        help = 'Path to store the created out files'
    )


    # Parse the arguments
    opt = parser.parse_args()
    mask_path  = opt.mask_path
    mask_ext   = opt.mask_ext
    out_path   = opt.out_path
    # xml_path   = opt.xml_path
    #mask_bounding_path   = '/media/htic/NewVolume1/murali/mitosis/mitotic_count/bounding_test'
    min_area = 100
    
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    else:
       shutil.rmtree(out_path)
       os.mkdir(out_path)

    mask_path_ext = os.path.join(mask_path , '*.' + mask_ext)
    mask_images_path = glob.glob(mask_path_ext)

    no_of_bounding_boxes = 0

    for each_mask_path in tqdm(mask_images_path):

        cnts_flag = 0
        area_flag = 0
        # print each_mask_path
        mask_name_with_ext = os.path.basename(each_mask_path)
        txt_name = mask_name_with_ext.replace('.'+mask_ext,'.txt')
        
        img_mask = cv2.imread(each_mask_path)
        

        img_mask_gray = cv2.cvtColor(img_mask,cv2.COLOR_BGR2GRAY)
        _,im_bw = cv2.threshold(img_mask_gray, 200, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        kernel = np.ones([5,5],np.uint8)
        # opening = cv2.morphologyEx(im_bw,cv2.MORPH_OPEN,kernel)
        dilation = cv2.dilate(im_bw,kernel,iterations=3)
        cnts = cv2.findContours(dilation, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]
        if(cnts):
            cnts_flag = 1
        #print(cnts_flag) 
        eps = 5


        # tree = et.parse('mask.xml')
        # root = tree.getroot()
        # filename = root.find('filename')
        # filename.text = mask_name_with_ext
        # size = root.find('size')


        height = 512
        width = 512
        
        for c in cnts:
            (x, y, w, h) = cv2.boundingRect(c)
            xmin = x - eps
            ymin = y - eps
            xmax   = x + w + eps
            ymax   = y + h + eps
            if(xmin>width):
                xmin = width 
            if(ymin>height):
                ymin = height 
            if(xmax>width):
                xmax = width 
            if(ymax>height):
                ymax = height 
            
            if(xmin<0):
                xmin = 0
            if(ymin<0):
                ymin = 0
            if(xmax<0):
                xmax = 0
            if(ymax<0):
                ymax = 0

            x_center = (x + w/2.0) / width
            y_center = (y + h/2.0) / height
            w = w / width
            h = h / height
              



        #     color   = (0,0,255)
        #     line_thickness = 2

            area_ = cv2.contourArea(c)
            if area_ > min_area:
        #         cv2.rectangle(img_mask, (xmin,ymin), (xmax ,ymax),color, line_thickness)
        #         obj = tree.find('object')
        #         node = et.Element('bndbox')
        #         obj.insert(1,node)
        #         # write_xml(root,xmin,ymin,xmax,ymax,width,height)
                
        #         no_of_bounding_boxes += 1
                  area_flag = 1
        # #Write to xml only for files which return a bounding box.        
        if(area_flag and cnts_flag):
            with open(os.path.join(out_path,txt_name)) as f:
                f.write(0," ",x_center," ",y_center," ",w," ",h)
                f.write('\n')
        #else:
        #    print(each_mask_path,area_flag,cnts_flag)
    
        #write_img(img_mask,os.path.join(mask_bounding_path,mask_name_with_ext))
        # break
        # break
    # display_img(img_mask)
    # print ("No of bounding boxes:",no_of_bounding_boxes)
