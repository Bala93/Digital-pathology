size=256
stride=183
whole_img_ext='bmp'
whole_mask_ext='bmp'
sample_mask_ext='jpg'
whole_mask_path='/media/htic/NewVolume1/murali/mitosis/mitotic_count/test_masks'
whole_img_path='/media/htic/NewVolume1/murali/mitosis/mitotic_count/test_images'
sample_save_path='/media/htic/NewVolume1/murali/mitosis/mitotic_count/test_images_sample'

#Create patch from whole image
#python create_patch.py --input_path=${whole_img_path} --output_path=${sample_save_path} --img_ext=${whole_img_ext} --stride=${stride} --img_size=${size} --input_type='image' --pad=1 --pad_mode='constant'

#Create patch from whole mask
#python create_patch.py --input_path=${whole_mask_path} --output_path=${sample_save_path}  --img_ext=${whole_mask_ext} --stride=${stride} --img_size=${size} --input_type='mask' --pad=1 --pad_mode='constant'

#Create pix2pix dataset from patches
#Note:Delete all inside 'test'
#cd /media/htic/NewVolume1/murali/mitosis/codes/Digital-pathology/codes
#python create_dataset_pix2pix.py

#combine A and B into AB
#cd /media/htic/NewVolume1/murali/pytorch-CycleGAN-and-pix2pix/datasets
#python combine_A_and_B.py --fold_A '/media/htic/NewVolume1/murali/mitosis/mitotic_segment/test/A' --fold_B '/media/htic/NewVolume1/murali/mitosis/mitotic_segment/test/B' --fold_AB '/media/htic/NewVolume1/murali/mitosis/mitotic_segment/test/AB' 

#Run test.py
cd /media/htic/NewVolume1/murali/pytorch-CycleGAN-and-pix2pix
sh test.sh
