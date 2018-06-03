import xml.etree.ElementTree as et
import glob
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import imutils
from tqdm import tqdm


def display_img(img):
    cv2.imshow('win',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return

def write_img(img,img_path):
    cv2.imwrite(img_path,img)
    return

if __name__ == "__main__":
    
    '''
    This works when there is no overlap between mask regions. TODO: See if the can be extended to overlapping regions also.
    Example : 
    python bounding_box_create.py --mask_path='/media/htic/NewVolume1/murali/mitosis/512_mask' --mask_ext='jpg' --xml_path='/media/htic/NewVolume1/murali/mitosis/512_xml'
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
        '--xml_path',
        required = True,
        type = str,
        help = 'Path to store the created xml files'
    )


    # Parse the arguments
    opt = parser.parse_args()
    mask_path  = opt.mask_path
    mask_ext   = opt.mask_ext
    xml_path   = opt.xml_path
    mask_bounding_path   = '/media/htic/NewVolume1/murali/mitosis/512_mask_bound'

    if not os.path.exists(xml_path):
        os.mkdir(xml_path)


    mask_path_ext = os.path.join(mask_path , '*.' + mask_ext)
    mask_images_path = glob.glob(mask_path_ext)

    for each_mask_path in tqdm(mask_images_path):

        # print each_mask_path
        mask_name_with_ext = os.path.basename(each_mask_path)
        xml_name = mask_name_with_ext.replace('.jpg','.xml')
        
        img_mask = cv2.imread(each_mask_path)
        img_mask_gray = cv2.cvtColor(img_mask,cv2.COLOR_BGR2GRAY)
        _,im_bw = cv2.threshold(img_mask_gray, 200, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        kernel = np.ones([5,5],np.uint8)
        opening = cv2.morphologyEx(im_bw,cv2.MORPH_OPEN,kernel)
        dilation = cv2.dilate(opening,kernel,iterations=1)
        cnts = cv2.findContours(dilation, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]
        for c in cnts:
            (x, y, w, h) = cv2.boundingRect(c)
            area_ = cv2.contourArea(c)
            if area_ > 100:
                cv2.rectangle(img_mask, (x, y), (x + w, y + h), (0, 0, 255), 2)
        
        write_img(img_mask,os.path.join(mask_bounding_path,mask_name_with_ext))
        # break
    # display_img(img_mask)

    '''
        # Xml file
        cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]
        
        # Open original file
        tree = et.parse('mask.xml')
        root = tree.getroot()

        filename = root.find('filename')
        filename.text = img_name
        
        # loop over the contours
        for c in cnts:
        # compute the bounding box of the contour and then draw the
        # bounding box on both input images to represent where the two
        # images differ
            (x, y, w, h) = cv2.boundingRect(c)
            #text = str(w*h)
            #print text
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.rectangle(img_mask, (x, y), (x + w, y + h), (0, 0, 255), 2)
            #cv2.putText(img, text, (x, h), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)
            #cv2.putText(img_mask, text,(x,h) , cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA) 
            obj = tree.find('object')
            node = et.Element('bndbox')
            obj.insert(1,node)
            xmin = et.Element('xmin')
            ymin = et.Element('ymin')
            xmax = et.Element('xmax')
            ymax = et.Element('ymax')
            bndbox = root.find('object').find('bndbox')
            bndbox.insert(1,xmin)
            bndbox.insert(1,ymin)
            bndbox.insert(1,xmax)
            bndbox.insert(1,ymax)
            
            xmin = x-w/2
            ymin = x-h/2
            xmax = w/2+x
            ymax = h/2+x
            
            if(xmin>540):
                xmin = 540
            if(ymin>540):
                ymin = 540
            if(xmax>540):
                xmax = 540
            if(ymax>540):
                ymax = 540
            
            if(xmin<0):
                xmin = 0
            if(ymin<0):
                ymin = 0
            if(xmax<0):
                xmax = 0
            if(ymax<0):
                ymax = 0
            
            bndbox.find('xmin').text = str(xmin)
            bndbox.find('ymin').text = str(ymin)
            bndbox.find('xmax').text = str(xmax)
            bndbox.find('ymax').text = str(ymax)

        
        #cv2.imwrite('0_0210_bb.png',img)
        # plt.imshow(img)
        # plt.show()
        # plt.imshow(img_mask)
        # plt.show()
        #smask_path = file.replace('.png','')+'_smask.png'
        #cv2.imwrite(smask_path,img_mask)

        # Write back to file
        #et.write('file.xml')
        print xml_path
        tree.write(xml_path)
    '''