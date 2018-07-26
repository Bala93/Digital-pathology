src_model_path='/media/htic/NewVolume3/Balamurali/mitosis_classification/mitosis_models'
src_val_path='/media/htic/NewVolume3/Balamurali/mitosis_classification/dataset/test'
img_ext='bmp'
src_csv_path='/media/htic/NewVolume3/Balamurali/mitosis_classification/mitosis_models'
cuda_no='0'

for model_name in 'resnet101' 'densenet201' 'resnet152' 'densenet169'   #'vgg16_bn' 'vgg19_bn' 'inception'   
   do 
   val_path=${src_val_path}/${color}
   model_path=${src_model_path}/${model_name}_${color}
   echo ${model_name}_${color}
   csv_path=${src_csv_path}/${model_name}_${color}
   python evaluate_custom.py --model_path ${model_path} --val_path ${val_path} --img_ext ${img_ext} --csv_path ${csv_path} --cuda_no ${cuda_no} --model_name ${model_name}
  done
