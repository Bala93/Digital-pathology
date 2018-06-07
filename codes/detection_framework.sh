# For images
#!/bin/bash
size=512
stride=64
whole_input_path='/media/htic/NewVolume1/murali/mitosis/mitotic_count/whole_images'
whole_mask_path='/media/htic/NewVolume1/murali/mitosis/mitotic_count/whole_mask'
sample_path='/media/htic/NewVolume1/murali/mitosis/mitotic_count'
# For images
python create_patch.py --input_path="${whole_input_path}" --output_path="${sample_path}" --img_ext='bmp' --stride="${stride}" --img_size="${size}" --input_type='image'
# For masks
python create_patch.py --input_path="${whole_mask_path}" --output_path="${sample_path}" --img_ext='bmp'  --stride=${stride} --img_size=${size} --input_type='mask'
# Evaluate and prepare folder
