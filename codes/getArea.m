function [Area,bw,binar] = getArea(imgpath)
image1=imread(imgpath);
size(image1);
image_find_datatype=islogical(image1)
if image_find_datatype==1
    image1=uint8(image1);
end
binar=imbinarize(image1);
bw=regionprops(binar,'Area','PixelList');
for i=1:size(bw,1)
    A=bw(i).Area;
    Area(i,1)=A;
end
end
