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

def transform_img(img,transform_type,rotation=0):
    
    if transform_type == 'fliplr':
        img = np.fliplr(img)
        # print ("Entering fliplr")

    if transform_type == 'flipud':
        img = np.flipud(img)
        # print ("Entering flipud")

    if transform_type == 'pad':
        img = np.pad(img,25,'symmetric')

    # if transform_type == 'rot15':
    #     rows,cols,_ = img.shape
    #     M = cv2.getRotationMatrix2D((cols/2,rows/2),15,1)
    #     img = cv2.warpAffine(img,M,(cols,rows))

    if transform_type == 'rot10':
        rows,cols,_ = img.shape
        M = cv2.getRotationMatrix2D((cols/2,rows/2),10,1)
        img = cv2.warpAffine(img,M,(cols,rows))

    if transform_type == 'rot30':
        rows,cols,_ = img.shape
        M = cv2.getRotationMatrix2D((cols/2,rows/2),30,1)
        img = cv2.warpAffine(img,M,(cols,rows))
    
    if transform_type == 'rot50':
        rows,cols,_ = img.shape
        M = cv2.getRotationMatrix2D((cols/2,rows/2),50,1)
        img = cv2.warpAffine(img,M,(cols,rows))
    
    if transform_type == 'rot-10':
        rows,cols,_ = img.shape
        M = cv2.getRotationMatrix2D((cols/2,rows/2),-10,1)
        img = cv2.warpAffine(img,M,(cols,rows))
    
    if transform_type == 'rot-30':
        rows,cols,_ = img.shape
        M = cv2.getRotationMatrix2D((cols/2,rows/2),-30,1)
        img = cv2.warpAffine(img,M,(cols,rows))
    
    if transform_type == 'rot-5':
        rows,cols,_ = img.shape
        M = cv2.getRotationMatrix2D((cols/2,rows/2),-5,1)
        img = cv2.warpAffine(img,M,(cols,rows))

    if transform_type == 'rot5':
        rows,cols,_ = img.shape
        M = cv2.getRotationMatrix2D((cols/2,rows/2),5,1)
        img = cv2.warpAffine(img,M,(cols,rows))


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
    
    if transform_type == 'rot10':
        x = np.array([[xmin,ymin,1],[xmin,ymax,1],[xmax,ymin,1],[xmax,ymax,1]])
        M = cv2.getRotationMatrix2D((1042,1042),-10,1)
        r = np.dot(M,x.T)
        xmin_ = np.min(r[0])
        ymin_ = np.min(r[1])
        xmax_ = np.max(r[0])
        ymax_ = np.max(r[1])

    
    if transform_type == 'rot30':
        x = np.array([[xmin,ymin,1],[xmin,ymax,1],[xmax,ymin,1],[xmax,ymax,1]])
        M = cv2.getRotationMatrix2D((1042,1042),-30,1)
        r = np.dot(M,x.T)
        xmin_ = np.min(r[0])
        ymin_ = np.min(r[1])
        xmax_ = np.max(r[0])
        ymax_ = np.max(r[1])


    if transform_type == 'rot5':
        x = np.array([[xmin,ymin,1],[xmin,ymax,1],[xmax,ymin,1],[xmax,ymax,1]])
        M = cv2.getRotationMatrix2D((1042,1042),-5,1)
        r = np.dot(M,x.T)
        xmin_ = np.min(r[0])
        ymin_ = np.min(r[1])
        xmax_ = np.max(r[0])
        ymax_ = np.max(r[1])
    
    if transform_type == 'rot-5':
        x = np.array([[xmin,ymin,1],[xmin,ymax,1],[xmax,ymin,1],[xmax,ymax,1]])
        M = cv2.getRotationMatrix2D((1042,1042),5,1)
        r = np.dot(M,x.T)
        xmin_ = np.min(r[0])
        ymin_ = np.min(r[1])
        xmax_ = np.max(r[0])
        ymax_ = np.max(r[1])


    if transform_type == 'rot-10':
        x = np.array([[xmin,ymin,1],[xmin,ymax,1],[xmax,ymin,1],[xmax,ymax,1]])
        M = cv2.getRotationMatrix2D((1042,1042),10,1)
        r = np.dot(M,x.T)
        xmin_ = np.min(r[0])
        ymin_ = np.min(r[1])
        xmax_ = np.max(r[0])
        ymax_ = np.max(r[1])

    if transform_type == 'rot-30':
        x = np.array([[xmin,ymin,1],[xmin,ymax,1],[xmax,ymin,1],[xmax,ymax,1]])
        M = cv2.getRotationMatrix2D((1042,1042),30,1)
        r = np.dot(M,x.T)
        xmin_ = np.min(r[0])
        ymin_ = np.min(r[1])
        xmax_ = np.max(r[0])
        ymax_ = np.max(r[1])

    if transform_type == 'rot-50':
        x = np.array([[xmin,ymin,1],[xmin,ymax,1],[xmax,ymin,1],[xmax,ymax,1]])
        M = cv2.getRotationMatrix2D((1042,1042),50,1)
        r = np.dot(M,x.T)
        xmin_ = np.min(r[0])
        ymin_ = np.min(r[1])
        xmax_ = np.max(r[0])
        ymax_ = np.max(r[1])


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

    image_yellow_path ='/media/htic/Balamurali/Endoscope_segment/polyp_mask'
    val_img_path    ='/media/htic/Balamurali/Endoscope_segment/polyp' 
    img_paths = glob.glob(val_img_path)
    label_path = '/media/htic/NewVolume1/murali/Object_detection/models/research/data/polyp_label_map.pbtxt'

    NUM_CLASSES = 1
    label_map = label_map_util.load_labelmap(label_path)
    categories = label_map_util.convert_label_map_to_categories(label_map,max_num_classes=NUM_CLASSES,use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    # Eval with transforms
    eval_transform_list =['none','fliplr','flipud']#'rot10','rot-10','rot5','rot-5',

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

            #area_thresh  = 1500
            #window_shape = (512,512)
            # whole_image_dim = (3500,3500,3)
            #whole_image_dim = (2084,2084,3)
            #stride =  393 #332 #265 #457  # check for maximum mitotic cell.
            
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
                    #whole_image_trans = transform_img(whole_image,each_trans)

                    #r_sample,g_sample,b_sample = obtain_tiles(whole_image_trans,window_shape,stride)

                    no_of_rows = r_sample.shape[0]
                    no_of_cols = r_sample.shape[1]

                    for row in tqdm(range(no_of_rows)):
                        for col in tqdm(range(no_of_cols)):
                        
                            image_np = get_sub_image(r_sample,g_sample,b_sample,row,col)
                            # image_np = transform.rescale(image_np,2,order=3,preserve_range=True)
                            
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

                                # Divide by 2
                                # xmin /=2
                                # ymin /=2
                                # xmax /=2
                                # ymax /=2

                                xmin = xmin + col * stride
                                ymin = ymin + row * stride
                                xmax = xmax + col * stride
                                ymax = ymax + row * stride

                                xmin,ymin,xmax,ymax = transform_coord(xmin,ymin,xmax,ymax,height,width,each_trans) 
                                
                                updated_metric_box.append([xmin,ymin,xmax,ymax])
                                
                            boxes +=  updated_metric_box
                            scores += metric_scores
                print (len(boxes))
                if(len(boxes)>0):   
                    np_boxes  = np.array(boxes)
                    np_scores = np.array(scores)
                    boxes,scores = non_max_suppression_fast(np_boxes,0.6,np_scores)
                    upd_boxes = []
                    upd_scores = []
                    for ind,(xmin,ymin,xmax,ymax) in enumerate(boxes):
                        if (xmax-xmin)*(ymax-ymin) > area_thresh:
                            upd_boxes.append([xmin,ymin,xmax,ymax])
                            upd_scores.append(scores[ind])

                    metric_json[file_name]['boxes'] = upd_boxes
                    metric_json[file_name]['scores'] = upd_scores

                    for xmin,ymin,xmax,ymax in upd_boxes:
                        #if (xmax-xmin)*(ymax-ymin) > area_thresh:
                        cv2.rectangle(gt_box_img,(xmin,ymin),(xmax,ymax),(255,0,0),5)
                        cv2.putText(gt_box_img,str((xmax-xmin)*(ymax-ymin)),(xmin,ymax),cv2.FONT_HERSHEY_COMPLEX,2,(255,0,0),2,cv2.LINE_AA)
                        # cv2.namedWindow('win',cv2.WINDOW_NORMAL)
                        # cv2.imshow('win',gt_box_img)
                        # cv2.waitKey(0)
                        # cv2.destroyAllWindows()
		    
                    cv2.imwrite(os.path.join(detection_out_path,file_name) ,gt_box_img)

            with open(predicted_json_path,'w') as f:
                 json.dump(metric_json,f)
