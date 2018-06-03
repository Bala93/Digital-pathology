import cv2
import numpy as np
import glob 
import os


def get_class_label(img_path):

    img = cv2.imread(img_path)
    imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # flattened = imgray.flatten()
    img_B = img[:,:,0]
    img_G = img[:,:,1]
    img_R = img[:,:,2]

    (thresh,imgray_bw) = cv2.threshold(imgray,16,255,cv2.THRESH_BINARY)
    (thresh,img_B_bw) = cv2.threshold(img_B,128,255,cv2.THRESH_BINARY)
    (thresh,img_G_bw) = cv2.threshold(img_G,128,255,cv2.THRESH_BINARY)
    (thresh,img_R_bw) = cv2.threshold(img_R,128,255,cv2.THRESH_BINARY)

    blue_count  = np.count_nonzero(img_B_bw.flatten())
    red_count   = np.count_nonzero(img_R_bw.flatten())
    green_count = np.count_nonzero(img_G_bw.flatten())
    black_count = np.count_nonzero(imgray_bw.flatten() == 0)

    total_colors = float(blue_count + red_count + green_count + black_count)
    # total_color_wo_black = blue_count + red_count + green_count
    black_content = black_count/total_colors
    # other_content = total_color_wo_black/total_colors
    other_colors  = [blue_count,green_count,red_count]

    '''
    black : 0 
    blue  : 1
    green : 2
    red   : 3
    '''

    if black_content > 0.95:
        class_label = 0
    else:
        class_label = 1 #other_colors.index(max(other_colors)) + 1    


    return class_label

if __name__  == "__main__":
    
    # 4 class. 95% -- black else higher value class.
    #img_base_path = '/media/balamurali/NewVolume2/Deep_network/mitosis/output_mask/size_512_stride_512/' + '*.jpg'
    img_base_path = '/media/htic/Seagate Backup Plus Drive/ICIAR/dataset/scale_16/mask/samples/size_512_stride_256/*.jpg'
    img_files = glob.glob(img_base_path)
    #csv_path  = '/media/balamurali/NewVolume2/Deep_network/mitosis/output_mask/size_512_stride_512/out.csv'
    csv_path = '/media/htic/Seagate Backup Plus Drive/ICIAR/dataset/scale_16/mask/samples/out.csv'

    with open(csv_path,'w') as f:

        for img_path in img_files:
            class_label = get_class_label(img_path)
            # print get_class_label(img_path)
            img_name = os.path.basename(img_path)
            line = img_name + ',' + str(class_label) + '\n'
            f.write(line)
            # break