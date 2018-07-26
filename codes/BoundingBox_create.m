function BoundingBox_create(image_path,mask_path,Result_image_path,Result_text_path)
image_ext='jpg';
mask_ext='jpg';
image_width=512;
image_height=512;
src_path1 = dir([image_path '/*.' image_ext]);
src_path2 = dir([mask_path '/*.' mask_ext]);

for i=1:size(src_path2,1)
    image_name=src_path2(i).name;
    image_name_ext=image_name(1:end-4);
    mask=imread(fullfile(mask_path,image_name));
    I=imread(fullfile(image_path,image_name));
    find_mask_datatype=islogical(mask);
    if find_mask_datatype==1
       mask=unit8(mask);
    end
    mask1=imbinarize(mask);
    sum_mask1=sum(mask1(:));
    if sum_mask1>=1
    bw=regionprops(mask1,'BoundingBox');
    text=[image_name_ext '.txt'];
    text1=strcat(fullfile(Result_text_path,text));
    fid=fopen(text1,'w');
    for i1=1:length(bw)
        bw1=bw(i1).BoundingBox;
        x=bw1(1);y=bw1(2);w=bw1(3);h=bw1(4);
%         w1=x+w;
%         w2=y+h;
%         x1=x+w/2;
%         y1=y+h/2;
         x1=x+w/2;
         y1=y+h/2;
        x2=x1/image_width;y2=y1/image_height;w1=w/image_width;h1=h/image_height;
        fprintf(fid,'0 %f %f %f %f \n',x2,y2,w1,h1);
    end
        fclose(fid);
        source_image=fullfile(image_path,image_name);
        copyfile(source_image,Result_image_path);   
    end
    
end

end
