src_train_path='/media/htic/NewVolume3/Balamurali/mitosis_classification/dataset/train'
src_val_path='/media/htic/NewVolume3/Balamurali/mitosis_classification/dataset/test'
src_save_path='/media/htic/NewVolume3/Balamurali/mitosis_classification/mitosis_models'
cuda_no=1
no_classes=2
is_pretrained=1

for model_name in 'resnet152' # 'resnet101' 'densenet201'  'densenet169'
  do
   train_path=${src_train_path}
   val_path=${src_val_path}
   save_path=${src_save_path}/${model_name}
   python classify_custom.py --train_path ${train_path} --val_path ${val_path} --save_path ${save_path} --cuda_no ${cuda_no} --model_name ${model_name} --no_classes ${no_classes} --is_pretrained ${is_pretrained}
  done