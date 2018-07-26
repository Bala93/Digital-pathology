import Augmentor
import argparse


if __name__ == "__main__":

    '''
    python augment_glaucoma.py --inp_img_path= --out_img_path= 
    '''

    parser = argparse.ArgumentParser('Augment images with input image,mask folder and output path')
    parser.add_argument(
        '--inp_img_path',
        required=True,
        type=str,
        help='provide path which contains images'
    )

    parser.add_argument(
        '--out_img_path',
        required = True,
        type = str,
        help = 'provide path to which images are saved'
    )

    parser.add_argument(
        '--no_samples',
        required = True,
        type = int,
        help = "No of samples"
    )

    opt  = parser.parse_args()
    path_to_img = opt.inp_img_path
    out_img_path = opt.out_img_path
    no_samples = opt.no_samples

    p = Augmentor.Pipeline(source_directory=path_to_img,output_directory=out_img_path)
    #p.ground_truth(path_to_mask)
    p.flip_left_right(probability=0.5)
    p.flip_top_bottom(probability=0.5)
    p.sample(no_samples) 