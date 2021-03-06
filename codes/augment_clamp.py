import Augmentor
import argparse


if __name__ == "__main__":

    '''
    python augment_mitosis.py
        --inp_img_path=/media/htic/NewVolume1/murali/mitosis/mitotic_count/image_size_512_stride_32_updated 
        --inp_mask_path=/media/htic/NewVolume1/murali/mitosis/mitotic_count/mask_size_512_stride_32_updated 
        --out_img_path=/media/htic/NewVolume1/murali/mitosis/mitotic_count/image_size_512_stride_32_updated 
        --out_mask_path=/media/htic/NewVolume1/murali/mitosis/mitotic_count/mask_size_512_stride_32_updated 
        --no_samples = 50

    python augment_mitosis.py
        --inp_img_path=/media/htic/Balamurali/GE_Project/Jun19-18/trainingSetImages-augmented/unknown/img 
        --inp_mask_path=/media/htic/Balamurali/GE_Project/Jun19-18/trainingSetImages-augmented/unknown/mask
        --out_img_path=/media/htic/Balamurali/GE_Project/Jun19-18/trainingSetImages-augmented/unknown/img
        --out_mask_path=/media/htic/Balamurali/GE_Project/Jun19-18/trainingSetImages-augmented/unknown/mask
        --no_samples = 50

    python augment_clamp.py --inp_img_path=/media/htic/Balamurali/GE_Project/Jun26-18/trainingSetImages/unknown/img --inp_mask_path=/media/htic/Balamurali/GE_Project/Jun26-18/trainingSetImages/unknown/mask --out_img_path=/media/htic/Balamurali/GE_Project/Jun26-18/trainingSetImages/unknown/img --out_mask_path=/media/htic/Balamurali/GE_Project/Jun26-18/trainingSetImages/unknown/mask --no_samples=5000
    '''

    parser = argparse.ArgumentParser('Augment images with input image,mask folder and output path')
    parser.add_argument(
        '--inp_img_path',
        required=True,
        type=str,
        help='provide path which contains images'
    )

    parser.add_argument(
        '--inp_mask_path',
        required=True,
        type=str,
        help='provide path which contains masks'
    )

    parser.add_argument(
        '--out_img_path',
        required = True,
        type = str,
        help = 'provide path to which images are saved'
    )

    parser.add_argument(
        '--out_mask_path',
        required = True,
        type = str,
        help = 'provide path to which masks are saved'
    )

    parser.add_argument(
        '--no_samples',
        required = True,
        type = int,
        help = "No of samples"
    )

    opt  = parser.parse_args()
    path_to_img = opt.inp_img_path
    path_to_mask = opt.inp_mask_path
    out_img_path = opt.out_img_path
    out_mask_path = opt.out_mask_path
    no_samples = opt.no_samples

    p = Augmentor.Pipeline(source_directory=path_to_img,output_directory=out_img_path,ground_truth_output_directory=out_mask_path)
    p.ground_truth(path_to_mask)
    p.rotate(probability=1,max_left_rotation=5,max_right_rotation=5)
    p.flip_left_right(probability=1)
    p.flip_top_bottom(probability=1)
    p.rotate_random_90(probability=1)
    p.sample(no_samples) 
