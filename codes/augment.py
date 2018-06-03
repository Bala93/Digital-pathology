import Augmentor

p=Augmentor.Pipeline('/media/htic/NewVolume1/murali/mitosis/512_image/')
p.ground_truth('/media/htic/NewVolume1/murali/mitosis/512_mask/')
p.rotate(probability=1,max_left_rotation=5,max_right_rotation=5)
p.flip_top_bottom(probability=1)
p.sample(50)