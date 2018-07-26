size=512
stride=32
whole_img_ext='bmp'
whole_mask_ext='bmp'
sample_mask_ext='jpg'
whole_mask_path='/media/htic/NewVolume1/murali/mitosis/mitotic_count/mitotic_non_mitotic/'
whole_img_path='/media/htic/NewVolume1/murali/mitosis/mitotic_count/whole_images'
sample_save_path='/media/htic/NewVolume1/murali/mitosis/mitotic_count'
image_aug_path='/media/htic/NewVolume1/murali/mitosis/mitotic_count/image_size_512_stride_32_updated'
mask_aug_path='/media/htic/NewVolume1/murali/mitosis/mitotic_count/mask_size_512_stride_32_classes_updated'
#Stain Normalize Whole Image
#matlab -nodisplay -nodesktop -r "stain_normalization('${whole_img_path}','${whole_img_path}/1.bmp');quit"

# Whole image
# python create_patch.py --input_path=${whole_img_path} --output_path=${sample_save_path} --img_ext=${whole_img_ext} --stride=${stride} --img_size=${size} --input_type='image'

# Whole mask
# python create_patch.py --input_path=${whole_mask_path} --output_path=${sample_save_path}  --img_ext=${whole_mask_ext} --stride=${stride} --img_size=${size} --input_type='mask'

# Remove unnecessary files including masks and images

sample_img_path=${sample_save_path}/image_size_${size}_stride_${stride}_classes
sample_mask_path=${sample_save_path}/mask_size_${size}_stride_${stride}_classes
# mask_generate:Remove masks which are cut mask_generate1:Remove the patch itself if a mask is cut in that patch.
# matlab -nodisplay -nodesktop -r "mask_generate('${whole_mask_path}','${sample_mask_path}','${sample_img_path}');quit"

# Do augmentation
#no_samples=1000
#python augment_mitosis.py --inp_img_path=${sample_save_path}/image_size_${size}_stride_${stride}_updated --inp_mask_path=${sample_save_path}/mask_size_${size}_stride_${stride}_updated --out_img_path=${sample_save_path}/image_size_${size}_stride_${stride}_updated --out_mask_path=${sample_save_path}/mask_size_${size}_stride_${stride}_updated --no_samples=${no_samples}
#matlab -nodisplay -nodesktop -r "mask_generate('${whole_mask_path}','${mask_aug_path}','${image_aug_path}');quit"
#find . -type f -print | awk -F/ 'length($NF) > 25' | xargs rm

# Create xml using created mask
#NOTE:Change width,height in mask.xml before running the below code.
python bounding_box_create_classes.py --mask_path='/media/htic/NewVolume1/murali/mitosis/mitotic_count/mask_size_512_stride_32_classes_updated' --mask_ext='jpg' --xml_path='/media/htic/NewVolume1/murali/mitosis/512_32_xml_classes'

# Move the created images and xmls to object detection folder
# cd /media/htic/NewVolume1/murali/Object_detection/models/research/images/
# echo "Test Print."
# rm * 
# cd ${sample_save_path}/image_size_${size}_stride_${stride}_updated_red_green
# echo "Copying Augmented Images..."
# cp * /media/htic/NewVolume1/murali/Object_detection/models/research/images/

# cd /media/htic/NewVolume1/murali/Object_detection/models/research/annotations/xmls
# rm *
# echo "Copying XMLs..."
# cd ${sample_save_path}/xml_size_${size}_stride_${stride}_updated/
# cp * /media/htic/NewVolume1/murali/Object_detection/models/research/annotations/xmls/

# echo "Creating TFRecord..."
########################## Create record files using image and mask files
# cd /media/htic/NewVolume1/murali/Object_detection/models/research
# python object_detection/dataset_tools/create_mitosis_tf_record.py --label_map_path=/media/htic/NewVolume1/murali/Object_detection/models/research/data/mitosis_label_map.pbtxt --data_dir=/media/htic/NewVolume1/murali/Object_detection/models/research --output_dir=/media/htic/NewVolume1/murali/Object_detection/models/research/data/mitosis/${size}_${stride}_red_green
    
# echo "Start Training..."
## Perform training
# cd /media/htic/NewVolume1/murali/Object_detection/models/research
# export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
# python object_detection/train.py --logtostderr --pipeline_config_path=/media/htic/NewVolume1/murali/Object_detection/models/research/models/model_mitosis/faster_rcnn_resnet101_mitosis.config --train_dir=/media/htic/NewVolume1/murali/Object_detection/models/research/models/model_mitosis/train/${size}_${stride}_red_green

# Create graph
# NOTE: Remove respective Graph Directory before execution.
# cd /media/htic/NewVolume1/murali/Object_detection/models/research
# export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
# ckpt_no=20723 #14252 #9249 #75603 #76570 67453 66359 65260 68534 9677  64162 
# python object_detection/export_inference_graph.py --input_type image_tensor --pipeline_config_path=models/model_mitosis/faster_rcnn_resnet101_mitosis.config --trained_checkpoint_prefix=models/model_mitosis/train/${size}_${stride}_red_green/model.ckpt-${ckpt_no} --output_directory=models/model_mitosis/graph/${size}_${stride}_red_green

##Stain Normalize Test Image

# Test data results 
# cd /media/htic/NewVolume1/murali/Object_detection/models/research
# export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
# min_score_thresh=0.50
# python infer_patch_wise_eval.py --model_file=models/model_mitosis/graph/${size}_${stride}_red_green/frozen_inference_graph.pb --result_path=/media/htic/NewVolume1/murali/mitosis/mitotic_count/results/512_32_red_green --thresh=${min_score_thresh}
# Evaluation 
#cd /media/htic/NewVolume1/murali/mitosis/codes/Digital-pathology/codes
# python calculate_mean_ap.py --json_path=/media/htic/NewVolume1/murali/mitosis/mitotic_count/yolo_rcnn_custom_anchor_out 
