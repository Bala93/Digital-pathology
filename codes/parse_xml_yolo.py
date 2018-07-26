import xml.etree.ElementTree as et
import glob
import os
from tqdm import tqdm

xml_path = glob.glob('/media/htic/NewVolume1/murali/mitosis/mitotic_count/xml_size_512_stride_128_updated/*.xml')
txt_save_path = '/media/htic/NewVolume2/yolo/darknet_multiple/darknet/mitosis_settings/train_512_128'

for each_xml_path in tqdm(xml_path):
    
    tree = et.parse(each_xml_path)
    root = tree.getroot()
    bndboxes = root.find('object').findall('bndbox')
    txt_name = os.path.join(txt_save_path,os.path.basename(each_xml_path).replace('xml','txt'))

    with open(txt_name,'w') as f:

        for bndbox in tqdm(bndboxes):    
            height = float(root.find('size').find('height').text)
            width = float(root.find('size').find('width').text)
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)

            w = xmax - xmin 
            h = ymax - ymin 

            x_center = (xmin + w/2.0) / width
            y_center = (ymin + h/2.0) / height
            w = w / width
            h = h / height
            
            coordinate = "0 {} {} {} {}\n".format(x_center,y_center,w,h)
            f.write(coordinate)                
        
    # break
