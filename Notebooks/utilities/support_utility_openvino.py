from openvino.inference_engine import IENetwork,IECore
import numpy as np
# from post_process_pixel import PixelLinkDecoder
import cv2,os,time
import logging as LOG
import pickle
import pandas as pd

#----------------------------------------------------------------------------------------------
class create_plot(object):
    def __init__(self):
        self.dcd = None
        self.coco_labels = None
        self.coco_label_color = None 
        self.color_box = None
        
    def create_bbox(self,frame,bbox_coords,box_color = (255,0,0),thickness=2):
        """
        This function will create bbox on the image
        
        """
        self.box_color = box_color
        self.thickness = thickness
        return cv2.rectangle(frame, tuple(bbox_coords[0]), tuple(bbox_coords[1]), self.box_color, self.thickness)

    def trim_frame_with_bboxes(self,frame,bboxes,offset = 0):
        """
        given an image this will trim the output as per bbox and threshold
        output is trimmed  frames list, with bbox along with offset
        """
        op = []
        updated_bbox = []
        for b in bboxes:
            [x1,y1,x2,y2] = b
            b =[max(x1-offset,0),max(y1-offset,0),min(x2+offset,frame.shape[1]),min(y2+offset,frame.shape[0])]
            updated_bbox.append(b)
            op.append(frame[b[1]:b[3],b[0]:b[2]])
        return op,updated_bbox

    def filter_result_bbox(self,frame,result,thresh=.4):
        """
        takes input of object detection frame and result 
        and return output bbox
        """
        initial_w,initial_h = frame.shape[1],frame.shape[0]
        res_filt =  result[np.where(result[:,:,:,2]>thresh)]
        res_filt = res_filt[np.min(res_filt,axis=1)>=0]
        class_ids = res_filt[:,1].astype(int)
        bboxes = np.multiply([[initial_w,initial_h,initial_w,initial_h]],(res_filt[:,3:])).astype('int')
        return bboxes
    def rotate_image(self,image, angle):
        """
        Arguements : 
           image : image to be rotated 
           angle : Angle of rotation in degree
        Returns : 
            rotated image
        """
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        return result
    def get_resolution_thresh(self,image):
        """
        Arguments:
            sample image for which resolution has to be calculated
        Returns:
            resolution threshold range as per given ROIs
        """
        cv2.namedWindow("preview",cv2.WINDOW_NORMAL)
        
        font = cv2.FONT_HERSHEY_SIMPLEX 
        org = (20, 20) 
        fontScale = 1
        color = (255, 0, 0) 
        thickness = 2
        image1 = cv2.putText(image.copy(), "draw MINIMUM sized ROI and press Enter/Space", org, font,  
                            fontScale, color, thickness, cv2.LINE_AA) 
        r1 = cv2.selectROI("preview",image1,False,False)
        image2 = cv2.putText(image.copy(), "draw MAXIMUM sized ROI and press Enter/Space", org, font,  
                            fontScale, color, thickness, cv2.LINE_AA) 
        r2 = cv2.selectROI("preview",image2,False,False)
        cv2.destroyAllWindows()
        return (r1[2]*r1[3])/(image.shape[0]*image.shape[1]),(r2[2]*r2[3])/(image.shape[0]*image.shape[1])


    def resolution_filter(self,res_filt,resolution_thresh_range=[.01,.1]):
        """
        resolution_thresh_range = percentage range of resolution to be filtered
        """

        a = (res_filt[:,6] - res_filt[:,4]) *(res_filt[:,5] - res_filt[:,3])
        return res_filt[((a>resolution_thresh_range[0] )& (a<resolution_thresh_range[1]))]

    def trim_frame_with_result(self,frame,result,threshold=.4,offset=0,return_results = False,resolution_thresh_range = None):
        """
        given an image this will trim the output as per bbox and threshold
        output is trimmed  frames list with bboxes(with offset)
        resolution_thresh_range = percentage range of resolution to be filtered

        """
        initial_w,initial_h = frame.shape[1],frame.shape[0]
        res_filt =  result[np.where(result[:,:,:,2]>threshold)]
        res_filt = res_filt[np.min(res_filt,axis=1)>=0]
        if resolution_thresh_range is not None:
            res_filt = self.resolution_filter(res_filt,resolution_thresh_range)
        bboxes = np.multiply([[initial_w,initial_h,initial_w,initial_h]],(res_filt[:,3:])).astype('int')

        op,updated_bbox = self.trim_frame_with_bboxes(frame,bboxes,offset)
        if return_results:
            return op,updated_bbox,res_filt
        return op,updated_bbox

    def res2bbox(self,result,shape,resolution_thresh=None,thresh = .5, return_res = True):
        """
            Arguments:
                result : result in original 1,1,batch,7 format
                shape : (width,height) format for original image
                thresh : detection threshold
                return_res : wheather to returned filtered results or not
            Returns:
                return res_filt(if return_res),bboxes
        """
        initial_w,initial_h = shape
        res_filt =  result[np.where(result[:,:,:,2]>thresh)]
        res_filt = res_filt[np.min(res_filt,axis=1)>=0]
        class_ids = res_filt[:,1].astype(int)
        if resolution_thresh is not None:
            res_filt = self.resolution_filter(res_filt,resolution_thresh_range=resolution_thresh)

        bboxes = np.multiply([[initial_w,initial_h,initial_w,initial_h]],(res_filt[:,3:])).astype('int')
        if return_res:
            return res_filt,bboxes
        return bboxes
        
    def create_bbox_with_text(self,frame,bbox_coords,text,font=cv2.FONT_HERSHEY_SIMPLEX,font_scale=1.5,font_thickness=1,box_color = (255,0,0), text_color = (0,0,0),create_highlight = True,highlight_color = (51,255,255)):
        """
        This function will create bbox with text and highlight on top of that text
        bbox_coords : ((x1,y1),(x2,y2)) opposite coordinates
        """
        self.font = font
        self.box_color = box_color
        self.text_color = text_color
        self.highlight_color = highlight_color
        self.font_scale = font_scale
        
        frame_op = self.create_bbox(frame.copy(),bbox_coords,box_color=box_color)
        if create_highlight:
            (text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=font_thickness)[0]
            text_x,text_y = bbox_coords[0][0],bbox_coords[0][1]
            box_coords = ((text_x, text_y), (min(text_x + text_width,frame.shape[1]) , max(0, text_y - text_height)))
            cv2.rectangle(frame_op, box_coords[0], box_coords[1], highlight_color, cv2.FILLED)
        cv2.putText(frame_op, text, bbox_coords[0], font, fontScale=font_scale, color=text_color, thickness=font_thickness)

        return frame_op
        
        
        
    def pixel_text_detection(self,frame,op_pixel_detection,box_color = (255,0,0),pixel_conf_threshold=.1,link_conf_threshold=.1):
        """
        draw text using from op of pixel image detection
        """
        
        if self.dcd is None:
            self.dcd = PixelLinkDecoder()
        b,a = op_pixel_detection
        op_frame = frame.copy()
        dcd.load(op_frame, a, b,pixel_conf_threshold=.1,link_conf_threshold=.1)
        dcd.decode()  # results are in dcd.bboxes
        for box in dcd.bboxes:
            cv2.drawContours(op_frame, [box], 0, box_color, 2)
        return op_frame
            
        
        
    def label_coco(self,frame,result,thresh=.4,resolution_thresh_range=None,\
                   font=cv2.FONT_HERSHEY_SIMPLEX,font_scale=1.5,font_thickness=1, text_color = (0,0,0),create_highlight = True,coco_labels = ['person','bicycle','car','motorcycle',\
                                                              'airplane','bus','train','truck','boat',\
                                                              'traffic light','fire hydrant','stop sign',\
                                                              'parking meter','bench','bird','cat','dog',\
                                                              'horse','sheep','cow','elephant','bear','zebra',\
                                                              'giraffe','backpack','umbrella','handbag','tie',\
                                                              'suitcase','frisbee','skis','snowboard','sports ball',\
                                                              'kite','baseball bat','baseball glove','skateboard',\
                                                              'surfboard','tennis racket','bottle','wine glass','cup',\
                                                              'fork','knife','spoon','bowl','banana','apple','sandwich',\
                                                              'orange','broccoli','carrot','hot dog','pizza','donut','cake',\
                                                              'chair','couch','potted plant','bed','dining table','toilet',\
                                                              'tv','laptop','mouse','remote','keyboard',\
                                                              'cell phone','microwave','oven','toaster','sink',\
                                                              'refrigerator','book','clock','vase','scissors','teddy bear',\
                                                              'hair drier','toothbrush']):
        """
        This will work for coco labels
        Will create bbox and text highlight
        """
        op_frame =  frame.copy()
        if self.coco_labels is None:
#             with open(coco_labels, 'r') as f:
#             self.coco_labels = [x.strip() for x in f]
            self.coco_labels = coco_labels
        if self.coco_label_color is None:   
            self.coco_label_color = np.random.randint(0,255,(len(coco_labels),3)).tolist()
        initial_w,initial_h = frame.shape[1],frame.shape[0]
        res_filt =  result[np.where(result[:,:,:,2]>thresh)]
        if resolution_thresh_range is not None:
            res_filt = self.resolution_filter(res_filt,resolution_thresh_range)
        res_filt = res_filt[np.min(res_filt,axis=1)>=0]
        class_ids = res_filt[:,1].astype(int)
        bboxes = np.multiply([[initial_w,initial_h,initial_w,initial_h]],(res_filt[:,3:])).astype('int')
        for idx,box in enumerate(bboxes):
            text = self.coco_labels[class_ids[idx]-1]
            bbox_color = self.coco_label_color[class_ids[idx]-1]
            #frame,bbox_coords,text,font=cv2.FONT_HERSHEY_SIMPLEX,font_scale=1.5,font_thickness=1,box_color = (255,0,0), text_color = (0,0,0),create_highlight = True,highlight_color = (51,255,255)
            op_frame = self.create_bbox_with_text(op_frame,((box[0],box[1]),(box[2],box[3])),text = text,box_color = bbox_color,\
                                                  text_color =text_color,font= font,font_scale=font_scale,font_thickness=font_thickness,\
                                                 create_highlight=create_highlight,highlight_color=bbox_color)  
        return op_frame 
    
    
    def label_obj_detection(self,frame, result, thresh = .4,labels = ['head','upper_body'],colors = None,\
                                              font=cv2.FONT_HERSHEY_SIMPLEX,font_scale=1.5,font_thickness=1,\
                                text_color = (0,0,0),create_highlight = True,resolution_thresh_range=None):
        """
        Generalized object detection labeling
        """
        op_frame = frame.copy()
        if self.color_box is None:
            if colors is None:
                colors = np.random.randint(0,255,(len(labels),3)).tolist()
            self.color_box = colors
        colors = self.color_box
        initial_w,initial_h = frame.shape[1],frame.shape[0]
        res_filt =  result[np.where(result[:,:,:,2]>thresh)]
        res_filt = res_filt[np.min(res_filt,axis=1)>=0]
        class_ids = res_filt[:,1].astype(int)
        
        if resolution_thresh_range is not None:
            res_filt = self.resolution_filter(res_filt,resolution_thresh_range)
        bboxes = np.multiply([[initial_w,initial_h,initial_w,initial_h]],(res_filt[:,3:])).astype('int')
        for idx,box in enumerate(bboxes):
            text = labels[class_ids[idx]-1]
            bbox_color = colors[class_ids[idx]-1]
            op_frame = self.create_bbox_with_text(op_frame,((box[0],box[1]),(box[2],box[3])),text=text,box_color = bbox_color, \
                                                  text_color =text_color,font= font,font_scale=font_scale,font_thickness=font_thickness,\
                                                 create_highlight=create_highlight,highlight_color=bbox_color)    
        return op_frame  
    
    
    def write_text(self,frame,text,offset_from_corner = 30,location='top-left',font=cv2.FONT_HERSHEY_SIMPLEX,font_scale=2,text_color = (255,255,255),font_thickness=1,create_highlight = True,highlight_color = (51,255,255)):
        """
        Write some text on top of image at a fix offset
        location = ['top-left','bottom-left','top-right','bottom-right']
        """
        frame_op = frame.copy()
        (text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=font_thickness)[0]
        if type(location)==[tuple,list]:
            coord_1 = location
            coord_2 = (coord_1[0]+text_width,coord_1[1]-text_height)
        elif location == 'top-right':
            coord_1 = (frame.shape[1]-offset_from_corner-text_width,offset_from_corner+text_height)
            coord_2 = (frame.shape[1]-offset_from_corner,offset_from_corner)
        elif location =='top-left':
            coord_1 = (offset_from_corner,offset_from_corner+text_height)
            coord_2 = (offset_from_corner+text_width,offset_from_corner)
        elif location == 'bottom-right':
            coord_2 = (frame.shape[1]-offset_from_corner,frame.shape[0]-offset_from_corner-text_height)
            coord_1 = (frame.shape[1]-text_width-offset_from_corner,frame.shape[0]-offset_from_corner)
        elif location =='bottom-left':
            coord_1 = (offset_from_corner,frame.shape[0]-offset_from_corner)
            coord_2 = (offset_from_corner+text_width,frame.shape[0]-offset_from_corner-text_height)
            
        
        if create_highlight :
            cv2.rectangle(frame_op, coord_1, coord_2, highlight_color, cv2.FILLED)
        cv2.putText(frame_op, text, coord_1, font, fontScale=font_scale, color=text_color, thickness=font_thickness)
        return frame_op
            
            
    
#-----------------------------------------------------------------------------------------------------------------------        
            
            
    
        
class async_infer(object):
    def __init__(self,num_requests=4,ie_network=None):
        """
        num_requests : shape of max_request
        check for output only when 
        attrs are other attributes of frame 

        """
        self.num_requests = num_requests
        self.frames_buffer = [0]*num_requests
        self.attrs = [0]*num_requests
        self.cursor_id = 0
        self.in_frame  = None
        self.frame_processed = 0
        if ie_network is None:
            self.ie = IECore()
        else:
            self.ie = ie_network
        print("Available Devices : ", self.ie.available_devices)

#         self.load_model()

    
    def callback(self):
        for id, req in enumerate(self.exec_net.requests):
            req.set_completion_callback(py_callback=callback, py_data=id)
        
    def load_model(self,model_path,device,DYN_BATCH_ENABLED=False,max_batch_dyn=10,input_format="NCHW"):
        """
        model_path : path of xml model file
        DYN_BATCH_ENABLED : for enabling dynamic batch(# https://docs.openvinotoolkit.org/latest/classie__api_1_1InferRequest.html#a7598a35081e9beb4a67175acb371dd3c)
        max_batch_dyn : max batch size for dynamic batching
        device : CPU, GPU, "MULTI:CPU,GPU","HETERO:CPU,GPU"
        input_format : "NCHW","NHWC"

        """
        self.device = device
        if device in ["GPU","CPU"]:
            print("OPTIMIZATION_CAPABILITIES for %s: "%self.device,self.ie.get_metric(metric_name="OPTIMIZATION_CAPABILITIES", device_name=self.device))

        self.model_xml = model_path
        self.model_bin = os.path.splitext(self.model_xml)[0] + ".bin"
        self.net = self.ie.read_network(model=self.model_xml, weights=self.model_bin)

        if DYN_BATCH_ENABLED:
            self.net.batch_size = max_batch_dyn
            self.ie.set_config({'DYN_BATCH_ENABLED': 'YES'},self.device)

        print("model inputs :", self.net.input_info.keys())
        print("model outputs : ", self.net.outputs.keys())
        self.input_blob = next(iter(self.net.input_info))
        self.out_blob = list(self.net.outputs.keys())
        self.exec_net = self.ie.load_network(self.net, self.device, num_requests=self.num_requests)
        self.input_format = input_format
        blob_shape = self.net.input_info[self.input_blob].tensor_desc.dims
        self.batch_size = blob_shape[self.input_format.index("N")]
        self.channel = blob_shape[self.input_format.index("C")]
        self.height = blob_shape[self.input_format.index("H")]
        self.width = blob_shape[self.input_format.index("W")]

        #     self.NCHW = True
        # if input_format=="NHWC":
        #     self.batch_size, self.height, self.width, self.channel = self.net.input_info[self.input_blob].tensor_desc.dims
        #     self.NCHW = False


        # self.__load_first_request__()
        # del net
        
    def __load_first_request__(self):  
        print('loading sample element')
        sample_frame = np.random.random((1080, 1920, 3))
        try:
            for i in range(self.num_requests):
                self.predict(sample_frame)
        except Exception as e:
            print(e)
        for i in range(self.num_requests):
            self.postprocess_op()
        self.cursor_id = 0
        self.frame_processed = 0






    def preprocess_frame(self,frame,return_res=False):
        """
        Arguements : 
           frame : frame that need to be preprocessed
           return_res :  if processed frame have to be returned
        Returns : 
            processed output
        """

        in_frame = cv2.resize(frame.copy(), (self.width, self.height))
        if self.input_format=="NCHW":
            in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
            in_frame = in_frame.reshape((self.batch_size, self.channel, self.height, self.width))
        if self.input_format=="NHWC":
            in_frame = in_frame.reshape((self.batch_size, self.height, self.width, self.channel))

        self.in_frame = in_frame
        if return_res:
            return in_frame
    

    def predict_sync(self,frame):
        """
        predict sync only
        return output_values
        """
        self.preprocess_frame(frame)
        self.exec_net.requests[0].wait()
        self.exec_net.requests[0].infer({self.input_blob: self.in_frame})
        return [self.exec_net.requests[0].output_blobs[node].buffer for node in self.out_blob ]      
        #         if self.start_infer :
#             if self.net.requests[cur_request_id].wait(-1) == 0 :
#                 self.output = [exec_net.requests[cur_request_id].outputs[node] for node in self.output_blob]
#         else :Sampad Mahapatra
#             if len(self.frames_buffer)>=self.num_requests:
#                 self.start_infer = True    

    def predict(self,frame,attr = None):
        """
        predict async only
        attr : attribute
        """
        self.preprocess_frame(frame)
        try:
            self.exec_net.requests[self.cursor_id].wait()
            self.exec_net.start_async(request_id=self.cursor_id, inputs={self.input_blob: self.in_frame})
            self.frames_buffer[self.cursor_id] = frame
            if attr is not None:
                self.attrs[self.cursor_id] = attr
            self.cursor_id = (self.cursor_id+1)% self.num_requests

            self.frame_processed+=1
        except Exception as e:
            print("withing preprocess_frame "+str(self.model_xml.split("/")[-1]),e)

        #         if self.start_infer :
#             if self.net.requests[cur_request_id].wait(-1) == 0 :
#                 self.output = [exec_net.requests[cur_request_id].outputs[node] for node in self.output_blob]
#         else :
#             if len(self.frames_buffer)>=self.num_requests:
#                 self.start_infer = True
    def preprocess_dyn(self,frame):
        """
        Arguements : 
           frame : input image
        Returns : 
            preprocessed output
        """
        in_frame = cv2.resize(frame.copy(), (self.width, self.height))
        if self.input_format=="NCHW":
            in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
            in_frame = in_frame.reshape((1, self.channel, self.height, self.width))
        if self.input_format=="NHWC":
            in_frame = in_frame.reshape((1, self.height, self.width, self.channel))
        return in_frame




    def predict_batch_dyn(self,input_batch):
        """
        Arguements : 
           input_batch : list images that are need to be predicted
        Returns : 
            batch output
        """

        infer_request = self.exec_net.requests[0]
        inputs_count = len(input_batch)
        if inputs_count==0:
            return 

        inputs = np.array([self.preprocess_dyn(frame) for frame in input_batch])
        infer_request.set_batch(inputs_count)
        infer_request.input_blobs[self.input_blob] = inputs
        infer_request.async_infer()
        infer_request.wait()
        result = [infer_request.output_blobs[node].buffer[:inputs_count] for node in self.out_blob]
        return result

    def reinit_model(self):
        self.cursor_id = 0 
        self.frame_processed = 0 

                
    def postprocess_op(self,request_id = None):
        """
        use request id 
        prediction only after completing one complete buffer
        return frame, attribute(if available) and output
        """
        if request_id is None:
            request_id = self.cursor_id
        # if (self.exec_net.requests[request_id].wait(-1) == 0 ):
        # while self.exec_net.requests[request_id].wait()!=0:
        #     time.sleep(.05)
        # print(self.exec_net.requests[request_id].wait())
        self.exec_net.requests[request_id].wait()
        self.output = [self.exec_net.requests[request_id].output_blobs[node].buffer for node in self.out_blob]
        op_frame = self.frames_buffer[request_id]
        attr = self.attrs[request_id]

        return op_frame,attr,self.output

    def predict_batch_async(self,frames):
        """
        list of frames for which async prediction have to be done
        """
        self.reinit_model() 

        op_res = []
        frame_processed = 0
        for idx, frame in enumerate(frames):
            #masked_image = np.multiply(image.copy(),cv2.fillPoly(np.zeros(image.shape,np.uint8),np.array([image_dict['roi']],'int'),[1,1,1]))
            try:
                self.predict(frame)
            except Exception as error:
                LOG.error("Error %s in  model consumer", error, exc_info=True)
            #pdb.set_trace()
            if (self.frame_processed % self.num_requests == 0) or (idx == len(frames) - 1):
                # either input data is consumed or  max batch request is reached
                elements_processed = idx - frame_processed + 1
                frame_processed = idx + 1
                # requesting output from 0 index
                
                for i in range(elements_processed):
                
                    try:
                        frame, attr, res = self.postprocess_op(request_id=i)
                        op_res.append(res)

                    except Exception as err:
                        LOG.error("Error in  model processing %s", err, exc_info=True)
        return op_res

