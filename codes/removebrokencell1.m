function [flag] = removebrokencell(bw2,Are,sample_mask_area)
Are_length_image=size(Are,1);
Area_length=size(sample_mask_area,1);
count=0;
for i1=1:size(bw2,1)
    s_mask3=bw2(i1).Area;
    for i2=1:size(Are,1)
        if s_mask3~=Are(i2,1)
            count=count+1;
        end
    end
   
end
 Final_count=Are_length_image-Area_length;
    flag=1;
    if count==Final_count
        flag=2;
    end
end