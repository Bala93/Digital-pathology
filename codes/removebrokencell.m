function [s_mask1] = removebrokencell(bw2,Are,s_mask1)
for i1=1:size(bw2,1)
    s_mask3=bw2(i1).Area;
    s_mask4=bw2(i1).PixelList;
    count=0;
    for i2=1:size(Are,1)
        if s_mask3==Are(i2,1)
            count=1;
        end
    end
    if count==0
        for i3=1:size(s_mask4,1)
            x=s_mask4(i3,1);
            y=s_mask4(i3,2);
            s_mask1(y,x)=0;
        end 
    end
end
end    