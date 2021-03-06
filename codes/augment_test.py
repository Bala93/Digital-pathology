import Augmentor
import argparse


if __name__ == "__main__":

    '''
    python augment_test.py --inp_img_path=/media/htic/NewVolume1/murali/mitosis/mitotic_count/test_images --inp_mask_path=/media/htic/NewVolume1/murali/mitosis/mitotic_count/test_masks --out_img_path=/media/htic/NewVolume1/murali/mitosis/mitotic_count/test_images_augmented --out_mask_path=/media/htic/NewVolume1/murali/mitosis/mitotic_count/test_image_yellow_augmented --no_samples=120
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
    p.flip_left_right(probability=0.8)
    p.flip_top_bottom(probability=0.8)
    p.random_brightness(0.5,0.5,1)
    p.random_contrast(0.5,0.5,1)
    p.sample(no_samples) 