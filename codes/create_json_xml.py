import xml.etree.ElementTree as ET
import os
import json
import glob
from tqdm import tqdm

if __name__ == "__main__":

    xml_path = '/media/balamurali/NewVolume2/IIT-HTIC/GE_Project/Dataset/Tensorflow/annotations/xmls/'
    json_path = '/media/balamurali/NewVolume2/IIT-HTIC/GE_Project/Dataset/Tensorflow/annotations/xmls/out.json'
    xml_files = glob.glob(xml_path + '*.xml')
    
    metrics_gt = {}
    
    for xml_file in tqdm(xml_files):
        # xml_file = os.path.join(xml_path,'1_3000.xml')
        tree = ET.parse(xml_file)
        root = tree.getroot()
        filename = root.find('filename').text2
        bndbox_list = []

        for bndbox in root.iter('bndbox'):
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            bndbox_list.append([xmin,ymin,xmax,ymax])

        metrics_gt[filename] = bndbox_list
        # print metrics_gt
        # break
    print metrics_gt.keys()
    with open(json_path,'w') as f:
        json.dump(metrics_gt,f)
     