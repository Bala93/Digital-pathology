function mask_generate(mask_whole_path,mask_sample_path,img_sample_path)
updated_mask_sample_path = [mask_sample_path '_updated'];
updated_img_sample_path = [img_sample_path '_updated'];

% mask_whole_path  = 'mask_whole_path\';
% mask_sample_path = 'mask_samplepath\';
% img_sample_path  = 'img_samplepath\';


mkdir(updated_mask_sample_path);
mkdir(updated_img_sample_path);
mask_whole_extension = 'bmp';
mask_sample_extension = 'jpg';
img_extension  = 'jpg';

mask_whole_files = dir([fullfile(mask_whole_path,['*' mask_whole_extension])]);    
mask_whole_files_total = size(mask_whole_files,1);

textprogressbar('Creating mask');
for whole_mask_count = 1: mask_whole_files_total
   textprogressbar(whole_mask_count)   % Whole mask
   mask_whole_name_with_ext = mask_whole_files(whole_mask_count).name; 
   mask_whole_name     = mask_whole_name_with_ext(1:end-4);
   whole_mask_path     = fullfile(mask_whole_files(whole_mask_count).folder,mask_whole_name_with_ext);
   [whole_mask_area,whole_mask_bw,~]    = getArea(whole_mask_path);
   
   % Sample mask
   mask_sample_files = dir(fullfile(mask_sample_path, [mask_whole_name '_*.' mask_sample_extension]));
   no_mask_sample_files = size(mask_sample_files,1);

   for each_sample_mask = 1:no_mask_sample_files
       textprogressbar(whole_mask_count);
       mask_folder = mask_sample_files(each_sample_mask).folder; 
       mask_name =  mask_sample_files(each_sample_mask).name;
       mask_path        = fullfile(mask_folder,mask_name); 
       mask_img = imread(mask_path);
       mask_img_sum = sum(mask_img(:));
       % This condition is to check whether the mask has any white blobs.
       if mask_img_sum > 0
           [sample_mask_area,sample_mask_bw,mask_img]=getArea(mask_path);
           mask_updated = removebrokencell(sample_mask_bw,whole_mask_area,mask_img);
           src_img_path  = fullfile(img_sample_path,mask_name); 
           src_mask_path = fullfile(mask_sample_path,mask_name);
           dest_img_path = fullfile(updated_img_sample_path,mask_name);
           dest_mask_path = fullfile(updated_mask_sample_path,mask_name);
           mask_updated_sum=sum(mask_updated(:));
           if mask_updated_sum>0
            % mask with white region are alone copied
            % image is copied while mask region is written as the mask is
            % updated i.e small cells are deleted.
            copyfile(src_img_path,dest_img_path);
            imwrite(mask_updated,dest_mask_path);
           end
       end
   end
   end
end
