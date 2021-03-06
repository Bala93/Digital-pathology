size=512
stride=32
whole_img_ext='bmp'
whole_mask_ext='bmp'
sample_mask_ext='jpg'
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
#no_samples=20000
#python augment_mitosis.py --inp_img_path=${sample_save_path}/image_size_${size}_stride_${stride}_updated --inp_mask_path=${sample_save_path}/mask_size_${size}_stride_${stride}_updated --out_img_path=${sample_save_path}/image_size_${size}_stride_${stride}_updated --out_mask_path=${sample_save_path}/mask_size_${size}_stride_${stride}_updated --no_samples=${no_samples}
#find . -type f -print | awk -F/ 'length($NF) > 25'

# Create xml using created mask
#python bounding_box_create.py --mask_path=${sample_save_path}/mask_resize_${size}_stride_${stride}_updated --mask_ext=${sample_mask_ext} --xml_path=${sample_save_path}/xml_resize_${size}_stride_${stride}

# Move the created images and xmls to object detection folder
#cd /media/htic/NewVolume1/murali/Object_detection/models/research/images/
#rm * 
#cd ${sample_save_path}/image_resize_${size}_stride_${stride}_updated
#cp * /media/htic/NewVolume1/murali/Object_detection/models/research/images/
#cd /media/htic/NewVolume1/murali/Object_detection/models/research/annotations/xmls
#rm *
#cd ${sample_save_path}/xml_resize__${size}_stride_${stride}/
#cp * /media/htic/NewVolume1/murali/Object_detection/models/research/annotations/xmls/

# Create record files using image and mask files
#cd /media/htic/NewVolume1/murali/Object_detection/models/research
#python object_detection/dataset_tools/create_mitosis_tf_record.py --label_map_path=/media/htic/NewVolume1/murali/Object_detection/models/research/data/mitosis_label_map.pbtxt --data_dir=/media/htic/NewVolume1/murali/Object_detection/models/research --output_dir=/media/htic/NewVolume1/murali/Object_detection/models/research/data/mitosis/resize_${size}_${stride}

# Perform training
#cd /media/htic/NewVolume1/murali/Object_detection/models/research
#export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
#python object_detection/train.py --logtostderr --pipeline_config_path=/media/htic/NewVolume1/murali/Object_detection/models/research/models/model_mitosis/faster_rcnn_resnet101_mitosis.config --train_dir=/media/htic/NewVolume1/murali/Object_detection/models/research/models/model_mitosis/train/resize_${size}_${stride}

# Create graph
#cd /media/htic/NewVolume1/murali/Object_detection/models/research
#export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
#ckpt_no=5277
#python object_detection/export_inference_graph.py --input_type image_tensor --pipeline_config_path=models/model_mitosis/faster_rcnn_resnet101_mitosis.config --trained_checkpoint_prefix=models/model_mitosis/train/resize_${size}_${stride}/model.ckpt-${ckpt_no} --output_directory=models/model_mitosis/graph/resize_${size}_${stride}

# Test data results 
cd /media/htic/NewVolume1/murali/Object_detection/models/research
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
min_score_thresh=0.5
python infer_patch_wise_eval.py --model_file=models/model_mitosis/graph/resize_${size}_${stride}/frozen_inference_graph.pb --result_path=${sample_save_path}/results/resize_${size}_${stride}  --thresh=${min_score_thresh}

# Evaluation 
#python calculate_mean_ap.py --json_path=${sample_save_path}/results/resize_${size}_${stride}
