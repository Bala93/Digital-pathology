function histogram_eq()
src_path = '/media/htic/NewVolume1/murali/mitosis/mitotic_count/test_images/';
dst_path = '/media/htic/NewVolume1/murali/mitosis/mitotic_count/test_histeg_images/';

for i =1:15
    inp_img_path = fullfile(src_path,[int2str(i) '.bmp']);
    out_img_path = fullfile(dst_path,[int2str(i) '.bmp']);
    inp_img = imread(inp_img_path);
    out_img = histeq(inp_img);
    imwrite(out_img,out_img_path);
end
end
