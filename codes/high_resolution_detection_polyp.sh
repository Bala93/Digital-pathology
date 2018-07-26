whole_img_ext='jpg'
whole_mask_ext='jpg'

whole_mask_path='/media/htic/Balamurali/Endoscope_segment/polyp_mask'
whole_img_path='/media/htic/Balamurali/Endoscope_segment/polyp'
sample_save_path='/media/htic/Balamurali/Endoscope_segment/'

# Create xml using created mask
#NOTE:Change width,height in mask.xml before running the below code.
# python bounding_box_create.py --mask_path=${sample_save_path}/polyp_mask --mask_ext=${sample_mask_ext} --xml_path=${sample_save_path}/xml 

# Move the created images and xmls to object detection folder
# cd /media/htic/NewVolume1/murali/Object_detection/models/research/images/
# echo "Test Print."
# rm * 
# cd ${sample_save_path}/polyp
# echo "Copying Augmented Images..."
# cp * /media/htic/NewVolume1/murali/Object_detection/models/research/images/

# cd /media/htic/NewVolume1/murali/Object_detection/models/research/annotations/xmls
# rm *
# echo "Copying XMLs..."
# cd ${sample_save_path}/xml
# cp * /media/htic/NewVolume1/murali/Object_detection/models/research/annotations/xmls/

# echo "Creating TFRecord..."
########################## Create record files using image and mask files
# cd /media/htic/NewVolume1/murali/Object_detection/models/research
# python object_detection/dataset_tools/create_polyp_tf_record.py --label_map_path=/media/htic/NewVolume1/murali/Object_detection/models/research/data/polyp_label_map.pbtxt --data_dir=/media/htic/NewVolume1/murali/Object_detection/models/research --output_dir=/media/htic/NewVolume1/murali/Object_detection/models/research/data/polyp
    
# echo "Start Training..."
## Perform training
# cd /media/htic/NewVolume1/murali/Object_detection/models/research
# export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
# python object_detection/train.py --logtostderr --pipeline_config_path=/media/htic/NewVolume1/murali/Object_detection/models/research/models/model_polyp/faster_rcnn_resnet101_polyp.config --train_dir=/media/htic/NewVolume1/murali/Object_detection/models/research/models/model_polyp/train

# Create graph
# NOTE: Remove respective Graph Directory before execution.
#cd /media/htic/NewVolume1/murali/Object_detection/models/research
#export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
#ckpt_no= 
#python object_detection/export_inference_graph.py --input_type image_tensor --pipeline_config_path=models/model_polyp/faster_rcnn_resnet101_polyp.config --trained_checkpoint_prefix=models/model_polyp/train/model.ckpt-${ckpt_no} --output_directory=models/model_polyp/graph

##Stain Normalize Test Image

# Test data results 
cd /media/htic/NewVolume1/murali/Object_detection/models/research
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
python infer_polyp.py --model_file /media/htic/NewVolume1/murali/Object_detection/models/research/models/model_polyp/graph/frozen_inference_graph.pb --input_path /media/htic/NewVolume3/Balamurali/cvcvideoclinicdbtestpart1/1 --inp_img_ext jpg --output_path /media/htic/Balamurali/Endoscope_segment/result  --label_file /media/htic/NewVolume1/murali/Object_detection/models/research/data/polyp_label_map.pbtxt

#cd /media/htic/NewVolume1/murali/mitosis/codes/Digital-pathology/codes
# python calculate_mean_ap.py --json_path=/media/htic/NewVolume1/murali/mitosis/mitotic_count/yolo_rcnn_custom_anchor_out 
