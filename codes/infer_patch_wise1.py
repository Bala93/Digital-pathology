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
from skimage.io import imread
import cv2

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

    img_sample = np.dstack((sample_r,sample_g,sample_b))

    return img_sample


        
if __name__ == "__main__":

    # Settings 
    model_path = '/media/htic/NewVolume1/murali/Object_detection/models/research/models/model_mitosis/graph/frozen_inference_graph.pb'
    #imgs_path = '/input/*.png'
    #val_imgs_list   = '/media/htic/NewVolume1/murali/yolo/darknet/sakthi_nuclei/nuclei/whole_data/output/val.txt'
    #val_img_path    = '/media/htic/NewVolume1/murali/yolo/darknet/sakthi_nuclei/nuclei/whole_data/images/*.jpg'
    val_img_path    = '/media/htic/NewVolume1/murali/mitosis/dataset/mitotic_count/*.bmp'
    #img_paths       = []
    #with open(val_imgs_list,'r') as f:
    #    img_names = f.readlines()
    #    for img in img_names:
    #        img_paths.append(os.path.join(val_img_path,img.strip() + '.jpg'))
  
    # imgs_path = '
    img_paths = glob.glob(val_img_path)
    # img_paths   = ['/media/htic/NewVolume1/murali/Object_detection/models/research/datasets/clamp/input/1_2135.jpg']
    label_path = '/media/htic/NewVolume1/murali/Object_detection/models/research/data/mitosis_label_map.pbtxt'
    detection_out_path = '/media/htic/NewVolume1/murali/mitosis/dataset/mitotic_count/results/'
    predicted_json_path = os.path.join(detection_out_path,'predicted_out.json')
    
    NUM_CLASSES = 1
    label_map = label_map_util.load_labelmap(label_path)
    categories = label_map_util.convert_label_map_to_categories(label_map,max_num_classes=NUM_CLASSES,use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

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
            count = 0        

            #####################
            # Patch wise split and evaluate
            ####################

            
            window_shape = (512,512)
            whole_image_dim = (2084,2084,3)
            stride = 393 # check for maximum mitotic cell.
            count = 0
            

            for in_img_path in tqdm(img_paths):
                file_name = os.path.basename(in_img_path)
                metric_json[file_name] = {}
                whole_image = imread(in_img_path)
                height,width,_ = whole_image.shape

                r_sample,g_sample,b_sample = obtain_tiles(whole_image,window_shape,stride)

                no_of_rows = r_sample.shape[0]
                no_of_cols = r_sample.shape[1]
          
                whole_img = np.zeros(whole_image_dim,dtype=np.uint8)
                boxes = []
                scores = []

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
                            min_score_thresh = 0.50
                        )

                        updated_metric_box = []

                        for coords in metric_box:
                            xmin,ymin,xmax,ymax = coords

                            xmin = xmin + col * stride
                            ymin = ymin + row * stride
                            xmax = xmax + col * stride
                            ymax = ymax + row * stride

                            area = (xmax - xmin) * (ymax - ymin)
                            updated_metric_box.append([xmin,ymin,xmax,ymax])
                            # if area > 100:
                            cv2.rectangle(whole_img,(xmin,ymin),(xmax,ymax),(0,255,0),3)

                        boxes +=  updated_metric_box
                        scores += metric_scores
                        
                
                        # print (metric_json)
                        # print (output_dict['detection_boxes'])
                        # print (output_dict['detection_classes'])
                        # print (output_dict['detection_scores'])

                        # print ("Plot")
                        # Plot Image
                        # img_cv2_bgr = cv2.cvtColor(image_np,cv2.COLOR_RGB2BGR)
                        # cv2.imwrite(detection_out_path + str(count) + '.jpg',img_cv2_bgr)
                        # count += 1
                        # whole_img
                        # cv2.imshow('win',img_cv2_bgr)
                        # cv2.waitKey(0)
                        # cv2.destroyAllWindows()
                print (boxes)
                boxes,scores = non_max_suppression_fast(np.array(boxes),0.6,np.array(scores))                        
                print (boxes)
                metric_json[file_name]['boxes'] = boxes
                metric_json[file_name]['scores'] = scores

                cv2.imwrite(detection_out_path + file_name ,whole_img)
                #break
            with open(predicted_json_path,'w') as f:
                 json.dump(metric_json,f)

            #####################
            ####################