# -*- coding: utf-8 -*-
"""
ICIAR2018 - Grand Challenge on Breast Cancer Histology Images
https://iciar2018-challenge.grand-challenge.org/home/
"""

import xml.etree.ElementTree as ET
import numpy as np
from scipy.misc import imsave
import cv2
import os
import openslide
from tqdm import tqdm


def findExtension(directory,extension='.xml'):
    files = []    
    for file in os.listdir(directory):
        if file.endswith(extension):
            files += [file]
    files.sort()
    return files

def fillImage(image, coordinates,color=255):
   cv2.fillPoly(image, coordinates, color=color)
   return image
 
def readXML(filename):
    
    tree = ET.parse(filename)
    
    root = tree.getroot()
    regions = root[0][1].findall('Region')
    
    
    
    
    pixel_spacing = float(root.get('MicronsPerPixel'))
    
    labels = []
    coords = []
    length = []
    area = []
    
    
    for r in regions:
        area += [float(r.get('AreaMicrons'))]
        length += [float(r.get('LengthMicrons'))]
        try:
            label = r[0][0].get('Value')
        except:
            label = r.get('Text')
        if 'benign' in label.lower():
            label = 1
        elif 'in situ' in label.lower():
            label = 2
        elif 'invasive' in label.lower():
            label = 3
        
        labels += [label]
        vertices = r[1]
        coord = []
        for v in vertices:
            x = int(v.get('X'))
            y = int(v.get('Y'))
            coord += [[x,y]]
    
        coords += [coord]

    return coords,labels,length,area,pixel_spacing

def saveImage(filename,image_size,coordinates,labels,sample=16):
    # sample = 4 denotes downsampling by factor 4.
    #red is 'benign', green is 'in situ' and blue is 'invasive'
    colors = [(0,0,0),(255,0,0),(0,255,0),(0,0,255)]
    
    img = np.zeros(image_size,dtype=np.uint8)
    
    for c,l in zip(coordinates,labels):
        img1 = fillImage(img,[np.int32(np.stack(c))],color=colors[l])
        img2 = img1[::sample,::sample,:]
    imsave(filename,img2)

if __name__=='__main__':
    
    folder_name = '/media/htic/Seagate Backup Plus Drive/ICIAR/WSI/' #path to the dataset folder
    files = findExtension(folder_name)
    store = []
    for file in tqdm(files):
        file_name = file[:-4]
        
        print('Reading scan',file_name)
        scan = openslide.OpenSlide(folder_name+file_name+'.svs')
        dims = scan.dimensions
        img_size = (dims[1],dims[0],3)
        print('Generating thumbnail')
        
        tree = ET.parse(folder_name+file)
        
        coords,labels,length,area,pixel_spacing = readXML(folder_name+file)
        store += [[coords,labels,length,area,pixel_spacing]]
        saveImage(folder_name+file_name+'.png',img_size,coords,labels)
        # break
