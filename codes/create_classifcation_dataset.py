import os
import shutil as sh
from tqdm import tqdm
# import numpy

'''
    The code helps in converting the csv file with filename and classlabel to folder structure.
'''

csv_path = '/media/htic/Seagate Backup Plus Drive/ICIAR/dataset/scale_16/mask/samples/out.csv'
dataset_create_path = '/media/htic/Seagate Backup Plus Drive/ICIAR/dataset/scale_16_train_Val/'
src_img_path  = '/media/htic/Seagate Backup Plus Drive/ICIAR/dataset/scale_16/img/samples/size_512_stride_256/'

train_val_path = os.path.join(dataset_create_path,'train-val')
# TOD: Create train and test folder
if not os.path.exists(train_val_path):
    os.mkdir(train_val_path)

# train_val_path = os.path.join(dataset_create_path,'train-val')

class_names = ['Normal','Tumor']

for each in class_names:
    class_path = os.path.join(train_val_path,each)
    if not os.path.exists(class_path):
        os.mkdir(class_path)

with open(csv_path,'r') as f:
    lines = f.readlines()
    for line in tqdm(lines):
        info = line.strip().split(',') 
        file_name = info[0]
        class_label = int(info[1])
        dst = os.path.join(train_val_path,class_names[class_label])
        src = os.path.join(src_img_path,file_name)
        sh.copy(src,dst)
        # print file_name,class_label 
        # break