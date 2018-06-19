import tensorflow as tf
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import glob
import os
from tqdm import tqdm
import json
import argparse


from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util



def load_image_into_numpy_array(image):
    (im_width,im_height)  = image.size
    return np.array(image.getdata()).reshape((im_height,im_width,3)).astype(np.uint8)
        
if __name__ == "__main__":

    parser = argparse.ArgumentParser('Infer images in a folder')
    parser.add_argument(
        '--model_file',
        required = True,
        type = str,
        help = 'provide model file' )

    parser.add_argument(
        '--input_path',
        required = True,
        type = str,
        help = 'provide path which contains input images')

    parser.add_argument(
        '--inp_img_ext',
        required = True,
        type = str,
        help = 'provide input image ext')

    parser.add_argument(
        'output_path',
        required = True,
        type = str,
        help = 'provide path to save images')

    parser.add_argument(
        'label_file',
        required = True,
        type = str,
        help = 'provide label_file')


    # Argument parsed and assigned 
    opt = parser.parse_args()
    model_file  = opt.model_file
    inp_img_ext = opt.inp_img_ext
    label_file  = opt.label_file
    output_path = opt.output_path 
    NUM_CLASSES = 1

    val_img_path = os.path.join(input_path ,'*.' + inp_img_ext)
    img_paths = glob.glob(val_img_path)
    
    label_map = label_map_util.load_labelmap(label_file)
    categories = label_map_util.convert_label_map_to_categories(label_map,max_num_classes=NUM_CLASSES,use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    # Initializing the graph
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(model_file,'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def,name='')

    with detection_graph.as_default():
        with tf.Session() as sess:
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            detection_fields = ['num_detections','detection_boxes','detection_scores','detection_classes']#,'detection_masks']
            
            for key in detection_fields:
                tensor_name = key + ':0'
                tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
            
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
            count = 0        

            for in_img_path in tqdm(img_paths):
                
                img_name = os.path.basename(in_img_path)

                out_img_path = os.path.join(detection_out_path,img_name)
                # Read Image
                image = Image.open(in_img_path)
                image_np = load_image_into_numpy_array(image)
                # print (image_np.shape)

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
		            min_score_thresh = 0.90
                )
                
                metric_json[img_name]['boxes'] = metric_box
                metric_json[img_name]['scores'] = metric_scores

                # Plot Image
                plt.figure(figsize=(12,8))
                plt.imshow(image_np)
                plt.savefig(out_img_path)
