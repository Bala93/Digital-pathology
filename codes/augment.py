import Augmentor

path = '/media/htic/Balamurali/Histopath/sakthi_final_2012_dataset_resized/train/mitotic'
count = 675
p=Augmentor.Pipeline(path)
#p.ground_truth('/media/htic/NewVolume1/murali/mitosis/512_mask/')
p.rotate(probability=1,max_left_rotation=5,max_right_rotation=5)
p.flip_top_bottom(probability=0.5)
p.flip_left_right(probability=0.5)
p.sample(count)
