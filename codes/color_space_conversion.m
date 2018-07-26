function color_space_conversion(input_path,color_space)

image_extension='jpg';
if color_space=='hsv'
HSV_color_space=[input_path '_HSV' '/'];
mkdir(HSV_color_space);
images=dir([input_path '/*' '.jpg']);
images_count=size(images,1)
for i=1:images_count
    image_name=images(i).name;
    image_name_ext=image_name(1:end-4);
    image=imread(fullfile([input_path '/' image_name]));
    hsv=rgb2hsv(image);
    imwrite(hsv,[HSV_color_space image_name]);
    

end
end
if color_space=='lab'
Lab_color_space=[input_path '_Lab' '/'];
mkdir(Lab_color_space);
images=dir([input_path '/*' '.jpg']);
images_count=size(images,1)
for i=1:images_count
    image_name=images(i).name;
    image_name_ext=image_name(1:end-4);
    image=imread(fullfile([input_path '/' image_name]));
    lab=rgb2lab(image);
    imwrite(lab,[Lab_color_space image_name]);
    

end
end
if color_space=='lin'
Lin_color_space=[input_path '_Lin' '/'];
mkdir(Lin_color_space);
images=dir([input_path '/*' '.jpg']);
images_count=size(images,1)
for i=1:images_count
    image_name=images(i).name;
    image_name_ext=image_name(1:end-4);
    image=imread(fullfile([input_path '/' image_name]));
    lin=rgb2lin(image);
    imwrite(lin,[Lin_color_space image_name]);

end
end
end
% if color_space=='lab'
% Lab_color_space=[output_path 'Lab'];
% mkdir(Lab_color_space);
% 

