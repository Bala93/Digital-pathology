import tensorflow as tf
from PIL import Image
import matplotlib
matplotlib.use('pdf')

from matplotlib import pyplot as plt
import numpy as np
import glob
import os
from tqdm import tqdm
import json
from skimage.util import view_as_windows
from skimage.io import imread,imsave
from skimage import transform
import cv2
import argparse

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from non_maximum_supression import non_max_suppression_fast


def load_image_into_numpy_array(image):
    (im_width,im_height)  = image.size
    return np.array(image.getdata()).reshape((im_height,im_width,3)).astype(np.uint8)


def obtain_tiles(img,window_shape,stride):

    r_channel = img[:,:,0]
    g_channel = img[:,:,1]
    b_channel = img[:,:,2]

    r_sample = view_as_windows(r_channel,window_shape,step=stride)
    g_sample = view_as_windows(g_channel,window_shape,step=stride)
    b_sample = view_as_windows(b_channel,window_shape,step=stride)

    return r_sample,g_sample,b_sample


def get_sub_image(r_sample,g_sample,b_sample,row,col):
    
    sample_r = r_sample[row,col]
    sample_g = g_sample[row,col]
    sample_b = b_sample[row,col]
    #Interpolation
    # sample_r = transform.rescale(sample_r,2,order=3,preserve_range=True)
    # sample_g = transform.rescale(sample_g,2,order=3,preserve_range=True)
    # sample_b = transform.rescale(sample_b,2,order=3,preserve_range=True)

    img_sample = np.dstack((sample_r,sample_g,sample_b))

    return img_sample

def transform_img(img,transform_type):
    
    if transform_type == 'fliplr':
        img = np.fliplr(img)
        # print ("Entering fliplr")

    if transform_type == 'flipud':
        img = np.flipud(img)
        # print ("Entering flipud")

    if transform_type == 'pad':
        img = np.pad(img,25,'symmetric')

    return img


def transform_coord(xmin,ymin,xmax,ymax,height,width,transform_type):


    if transform_type == 'fliplr':
        xmax_ = width - xmin
        xmin_ = width - xmax
        ymax_ = ymax
        ymin_ = ymin


    if transform_type == 'flipud':
        
        ymax_ = height - ymin
        ymin_ = height - ymax
        xmin_ = xmin
        xmax_ = xmax

    if transform_type == 'none':
        xmin_,ymin_,xmax_,ymax_ = xmin,ymin,xmax,ymax

    
    if transform_type == 'pad':
        xmin_ = xmin - 25
        xmax_ = xmax - 25
        ymin_ = ymin - 25
        ymax_ = ymax - 25



    return xmin_,ymin_,xmax_,ymax_


        
if __name__ == "__main__":
    
    # os.environ["CUDA_VISIBLE_DEVICES"] = '1'

    # Settings 

    parser = argparse.ArgumentParser('Testing the model with test dataset')

    parser.add_argument(
        '--model_file',
        required = True,
        type = str,
        help = 'The graph file path'
    )

    parser.add_argument(
        '--result_path',
        required = True,
        type = str,
        help = 'Path to which results should be saved'
    )

    parser.add_argument(
        '--thresh',
        required = True,
        type = float,
        help = 'Min score threshold'
    )

    opt = parser.parse_args()
    model_path = opt.model_file
    detection_out_path = opt.result_path

    if not os.path.exists(detection_out_path):
        os.mkdir(detection_out_path)


    predicted_json_path = os.path.join(detection_out_path,'predicted_out.json')
    min_score_thresh = opt.thresh

    image_yellow_path = '/media/htic/NewVolume1/murali/mitosis/mitotic_count/test_image_yellow'
    val_img_path    = '/media/htic/NewVolume1/murali/mitosis/mitotic_count/test_images/*.bmp'
    img_paths = glob.glob(val_img_path)
    label_path = '/media/htic/NewVolume1/murali/Object_detection/models/research/data/mitosis_label_map.pbtxt'

    NUM_CLASSES = 1
    label_map = label_map_util.load_labelmap(label_path)
    categories = label_map_util.convert_label_map_to_categories(label_map,max_num_classes=NUM_CLASSES,use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    # Eval with transforms
    eval_transform_list = ['none','fliplr','flipud']


    # Initializing the graph
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(model_path,'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def,name='')

    with detection_graph.as_default():
        with tf.Session() as sess:
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            # print (all_tensor_names)
            #######
            tensor_dict = {}
            detection_fields = ['num_detections','detection_boxes','detection_scores','detection_classes']#,'detection_masks']
            
            for key in detection_fields:
                tensor_name = key + ':0'
                tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
            
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
            metric_json = {}

            #####################
            # Patch wise split and evaluate
            ####################

            
            window_shape = (256,256)
            whole_image_dim = (2084,2084,3)
            stride = 457 #393 # check for maximum mitotic cell.
            
            # augmented_whole_image_box = {}
            #count = 0          
            for in_img_path in tqdm(img_paths):
                file_name = os.path.basename(in_img_path)
                metric_json[file_name] = {}
                whole_image = imread(in_img_path)
                height,width,_ = whole_image_dim
                # bounding_box_img = np.zeros(whole_image_dim,dtype=np.uint8) 
                gt_box_img_path = os.path.join(image_yellow_path,file_name.replace('.bmp','.jpg'))
                gt_box_img = cv2.imread(gt_box_img_path)

                boxes = []
                scores = []

                for each_trans in tqdm(eval_transform_list):
                    whole_image_trans = transform_img(whole_image,each_trans)

                    r_sample,g_sample,b_sample = obtain_tiles(whole_image_trans,window_shape,stride)

                    no_of_rows = r_sample.shape[0]
                    no_of_cols = r_sample.shape[1]

                    for row in tqdm(range(no_of_rows)):
                        for col in tqdm(range(no_of_cols)):
                        
                            image_np = get_sub_image(r_sample,g_sample,b_sample,row,col)
                            output_dict  = sess.run(tensor_dict,feed_dict={image_tensor:np.expand_dims(image_np,0)})

                            output_dict['num_detections']    = int(output_dict['num_detections'][0])
                            output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
                            output_dict['detection_boxes']   = output_dict['detection_boxes'][0]
                            output_dict['detection_scores']  = output_dict['detection_scores'][0]

                            # Visualize the bounding box
                            _,metric_box,metric_scores = vis_util.visualize_boxes_and_labels_on_image_array(
                                image_np,
                                output_dict['detection_boxes'],
                                output_dict['detection_classes'],
                                output_dict['detection_scores'],
                                category_index,
                                line_thickness=5,
                                use_normalized_coordinates=True,
                                min_score_thresh = min_score_thresh
                            )
                            #print(metric_box)
                            updated_metric_box = []

                            for coords in metric_box:
                                xmin,ymin,xmax,ymax = coords

                                xmin = xmin + col * stride
                                ymin = ymin + row * stride
                                xmax = xmax + col * stride
                                ymax = ymax + row * stride

                                xmin,ymin,xmax,ymax = transform_coord(xmin,ymin,xmax,ymax,height,width,each_trans) 

                                updated_metric_box.append([xmin,ymin,xmax,ymax])

                            boxes +=  updated_metric_box
                            scores += metric_scores
                            
                np_boxes  = np.array(boxes)
                np_scores = np.array(scores)
                boxes,scores = non_max_suppression_fast(np_boxes,0.5,np_scores)
                metric_json[file_name]['boxes'] = boxes
                metric_json[file_name]['scores'] = scores

                [cv2.rectangle(gt_box_img,(xmin,ymin),(xmax,ymax),(0,0,0),5)for xmin,ymin,xmax,ymax in boxes]
                    # cv2.namedWindow('win',cv2.WINDOW_NORMAL)
                    # cv2.imshow('win',gt_box_img)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                cv2.imwrite(os.path.join(detection_out_path,file_name) ,gt_box_img)

            with open(predicted_json_path,'w') as f:
                 json.dump(metric_json,f)
