size=512
stride=32
whole_img_ext='bmp' 
whole_mask_ext='bmp'
whole_mask_path='/media/htic/NewVolume1/murali/mitosis/mitotic_count/whole_mask'
whole_img_path='/media/htic/NewVolume1/murali/mitosis/mitotic_count/whole_images'
sample_save_path='/media/htic/NewVolume1/murali/mitosis/mitotic_count'

# Whole image
#python create_patch.py --input_path=${whole_img_path} --output_path=${sample_save_path} --img_ext=${whole_img_ext} --stride=${stride} --img_size=${size} --input_type='image'

# Sample image
#python create_patch.py --input_path=${whole_mask_path} --output_path=${sample_save_path}  --img_ext=${whole_mask_ext} --stride=${stride} --img_size=${size} --input_type='mask'

# Remove unnecessary files including masks and images

sample_img_path=${sample_save_path}/image_size_${size}_stride_${stride}
sample_mask_path=${sample_save_path}/mask_size_${size}_stride_${stride}
#matlab -nodisplay -nodesktop -r "mask_generate('${whole_mask_path}','${sample_mask_path}','${sample_img_path}');quit"

# Do augmentation 
python augment_mitosis.py --inp_img_path=${sample_save_path}/image_size_${size}_stride_${stride}_updated --inp_mask_path=${sample_save_path}/mask_size_${size}_stride_${stride}_updated --out_img_path=${sample_save_path}/image_size_${size}_stride_${stride}_updated --out_mask_path=${sample_save_path}/mask_size_${size}_stride_${stride}_updated


