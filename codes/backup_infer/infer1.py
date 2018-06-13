import tensorflow as tf
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import glob
import os
from tqdm import tqdm
import json


from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util



def load_image_into_numpy_array(image):
    (im_width,im_height)  = image.size
    return np.array(image.getdata()).reshape((im_height,im_width,3)).astype(np.uint8)

# def process_image(detection_graph,image_np):
#     with detection_graph.as_default():
#         with tf.Session() as sess:
#             ops = tf.get_default_graph().get_operations()
#             all_tensor_names = {output.name for op in ops for output in op.outputs}
#             # print (all_tensor_names)

#             #######
#             tensor_dict = {}
#             detection_fields = ['num_detections','detection_boxes','detection_scores','detection_classes']#,'detection_masks']
            
#             for key in detection_fields:
#                 tensor_name = key + ':0'
#                 tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
            
#             image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

#             output_dict  = sess.run(tensor_dict,feed_dict={image_tensor:np.expand_dims(image_np,0)})

#             output_dict['num_detections']    = int(output_dict['num_detections'][0])
#             output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
#             output_dict['detection_boxes']   = output_dict['detection_boxes'][0]
#             output_dict['detection_scores']  = output_dict['detection_scores'][0]

#         # print (output_dict['detection_boxes'].shape,output_dict['detection_scores'].shape)
#     return output_dict

        
if __name__ == "__main__":

    # Settings 
    model_path = '/media/htic/NewVolume1/murali/Object_detection/models/research/models/model_mitosis/graph/frozen_inference_graph.pb'
    #imgs_path = '/input/*.png'
    #val_imgs_list   = '/media/htic/NewVolume1/murali/yolo/darknet/sakthi_nuclei/nuclei/whole_data/output/val.txt'
    #val_img_path    = '/media/htic/NewVolume1/murali/yolo/darknet/sakthi_nuclei/nuclei/whole_data/images/*.jpg'
    val_img_path    = '/media/htic/NewVolume1/murali/Object_detection/models/research/images/*.jpg'
    #img_paths       = []
    #with open(val_imgs_list,'r') as f:
    #    img_names = f.readlines()
    #    for img in img_names:
    #        img_paths.append(os.path.join(val_img_path,img.strip() + '.jpg'))
  
    # imgs_path = '
    img_paths = glob.glob(val_img_path)
    # img_paths   = ['/media/htic/NewVolume1/murali/Object_detection/models/research/datasets/clamp/input/1_2135.jpg']
    label_path = '/media/htic/NewVolume1/murali/Object_detection/models/research/data/mitosis_label_map.pbtxt'
    detection_out_path = '/media/htic/NewVolume1/murali/Object_detection/models/research/images/results'
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

            # Do the same for video file. 
            for in_img_path in tqdm(img_paths):
                
                img_name = os.path.basename(in_img_path)
                metric_json[img_name] = {}

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
                # print (metric_json)
                # print (output_dict['detection_boxes'])
                # print (output_dict['detection_classes'])
                # print (output_dict['detection_scores'])

                # Plot Image
                plt.figure(figsize=(12,8))
                plt.imshow(image_np)
                plt.savefig(out_img_path)
                #plt.show()
                # break
                # metric_json = json.dumps(metric_json)
                # if count > 5:
                #     break
                # count += 1

            with open(predicted_json_path,'w') as f:
                 json.dump(metric_json,f)
