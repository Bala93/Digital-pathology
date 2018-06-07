import tensorflow as tf
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import glob
import os
from tqdm import tqdm
import json
import urllib.request as urllib
import cv2

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util



def load_image_into_numpy_array(image):
    (im_width,im_height)  = image.size
    return np.array(image.getdata()).reshape((im_height,im_width,3)).astype(np.uint8)

        
if __name__ == "__main__":

    # Settings 
    # print "IN"
    model_path = '/media/htic/NewVolume1/murali/Object_detection/models/research/models/model_clamp/graph/frozen_inference_graph.pb'
    label_path = '/media/htic/NewVolume1/murali/Object_detection/models/research/data/clamp_label_map.pbtxt'
    detection_out_path = '/media/htic/NewVolume1/murali/Object_detection/models/research/datasets/clamp/output'
    predicted_json_path = os.path.join(detection_out_path,'predicted_out.json')
    url = 'http://172.16.101.221:8080/shot.jpg'

    
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
            frame_count = 0

            while (1):

                # Do the same for video file.
                imgresp = urllib.urlopen(url)
                imgnp  = np.array(bytearray(imgresp.read()),dtype=np.uint8)
                img_cv2_bgr = cv2.imdecode(imgnp,-1)
                image_np_whole = cv2.cvtColor(img_cv2_bgr,cv2.COLOR_BGR2RGB)
                height,width,_ = image_np_whole.shape
                diff_  = width - height
                scale_ = int(diff_ / 2)
                end_   = width - scale_;
                image_np_resized = image_np_whole[:,scale_:end_,:]
            
                # image = 
                # img_name = '{%.3d.j}'.format(frm)
                img_name = '1.jpg'
                metric_json[img_name] = {}
(
                # out_img_path = os.path.join(detection_out_path,img_name)
                # Read Image
                # image = Image.open(in_img_path)
                # image_np = load_image_into_numpy_array(image_np)

                output_dict  = sess.run(tensor_dict,feed_dict={image_tensor:np.expand_dims(image_np_resized,0)})
                #TODO: Do conversion to image_np_whole space by just adding scale_ 
                output_dict['num_detections']    = int(output_dict['num_detections'][0])
                output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
                output_dict['detection_boxes']   = output_dict['detection_boxes'][0]
                output_dict['detection_scores']  = output_dict['detection_scores'][0]


                # Visualize the bounding box
                _,metric_box,metric_scores = vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np_whole,
                    output_dict['detection_boxes'],
                    output_dict['detection_classes'],
                    output_dict['detection_scores'],
                    category_index,
                    shift_right = scale_,
                    line_thickness=8,
                    use_normalized_coordinates=True,
                )
                
                metric_json[img_name]['boxes'] = metric_box
                metric_json[img_name]['scores'] = metric_scores
                
                # print (image_np_whole.shape)
                img_cv2_bgr = cv2.cvtColor(image_np_whole,cv2.COLOR_BGR2RGB)
                # print (img_cv2_bgr.shape)
                cv2.imshow('frame',img_cv2_bgr)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            #with open(predicted_json_path,'w') as f:
            #    json.dump(metric_json,f)