import xml.etree.ElementTree as et

def write_xml(root,xmin,ymin,xmax,ymax,width,height):
    
    xmin_tag = et.Element('xmin')
    ymin_tag = et.Element('ymin')
    xmax_tag = et.Element('xmax')
    ymax_tag = et.Element('ymax')
    objects = root.findall('object')
    
    for object in objects:
        bndbox = object.find('bndbox')
        bndbox.insert(1,xmin_tag)
        bndbox.insert(1,ymin_tag)
        bndbox.insert(1,xmax_tag)
        bndbox.insert(1,ymax_tag)
        
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
        
        bndbox.find('xmin').text = str(xmin)
        bndbox.find('ymin').text = str(ymin)
        bndbox.find('xmax').text = str(xmax)
        bndbox.find('ymax').text = str(ymax)        

    return

if __name__ == "__main__":
    tree = et.parse('mask_classes.xml')
    root = tree.getroot()
    filename = root.find('filename')
    filename.text = 'test'
    size = root.find('size')


    height = int(size.find('height').text)
    width = int(size.find('width').text)

    objs = root.findall('object')
    for obj in objs:
        name = obj.find('name')
        if(name.text=="Mitotic"):
            print(1)
        if(name.text=="NonMitotic"):
            print(2)



