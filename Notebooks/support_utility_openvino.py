from openvino.inference_engine import IENetwork, IEPlugin
import numpy as np
# from post_process_pixel import PixelLinkDecoder
import cv2,os,time

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
        return cv2.rectangle(frame, bbox_coords[0], bbox_coords[1], self.box_color, self.thickness)

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
            
        
        
    def label_coco(self,frame,result,thresh=.4,coco_labels = ['person','bicycle','car','motorcycle',\
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
                                                              'hair drier','toothbrush'],\
                   font=cv2.FONT_HERSHEY_SIMPLEX,font_scale=1.5,font_thickness=1, text_color = (0,0,0),create_highlight = True):
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
                                text_color = (0,0,0),create_highlight = True,resolution_thresh=None):
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
        bboxes = np.multiply([[initial_w,initial_h,initial_w,initial_h]],(res_filt[:,3:])).astype('int')
        if resolution_thresh is not None:
            bboxes = [b for b in bboxes if (b[3]-b[1])*(b[2]-b[0])>resolution_thresh]
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
    def __init__(self,buffer_shape=4):
        """
        buffer_shape : shape of max_request
        check for output only when 
        attrs are other attributes of frame 
        """
        self.buffer_shape = buffer_shape
        self.frames_buffer = [0]*buffer_shape
        self.attrs = [0]*buffer_shape
        self.cursor_id = 0
        self.in_frame  = None
        self.frame_processed = 0
#         self.load_model()

    
    def callback(self):
        for id, req in enumerate(self.exec_net.requests):
            req.set_completion_callback(py_callback=callback, py_data=id)
        
    def load_model(self,model_path,device,cpu_exension_path=None,gpu_extension_path=None,DYN_BATCH_ENABLED=False,max_batch_dyn=10):
        """
        model_path : path of xml model file
        cpu_extension_path : path of cpu extension .so file
        gpu_extension_path : gpu extension path that will be required to push model to intel-gpu
        DYN_BATCH_ENABLED : for enabling dynamic batch(# https://docs.openvinotoolkit.org/latest/classie__api_1_1InferRequest.html#a7598a35081e9beb4a67175acb371dd3c)
        max_batch_dyn : max batch size for dynamic batching
        """
        self.model_xml = model_path
        self.model_bin = os.path.splitext(self.model_xml)[0] + ".bin"
        self.Device = device
        self.cpu_exension_path =cpu_exension_path
        net = IENetwork(model=self.model_xml, weights=self.model_bin)
        try : 
            self.plugin = IEPlugin(device=self.Device)

        except Exception as e:
            print("this "+ device + " Not available")
            print(e)
        if device=='CPU' and cpu_exension_path is not None:
            self.plugin.add_cpu_extension(self.cpu_exension_path)
        if device=='GPU' and gpu_extension_path is not None:
            self.plugin.set_config({"CONFIG_FILE": gpu_extension_path})
        if DYN_BATCH_ENABLED:
            self.plugin.set_config({'DYN_BATCH_ENABLED': 'YES'})
            net.batch_size = max_batch_dyn

        print("model inputs :", net.inputs)
        print("model outputs : ", net.outputs)
        self.input_blob = next(iter(net.inputs))
        self.out_blob = list(net.outputs.keys())
        self.batch_size, self.channel, self.height, self.width = net.inputs[self.input_blob].shape
        self.exec_net = self.plugin.load(network=net, num_requests=self.buffer_shape)
        self.__load_first_request__()
        del net
        
    def __load_first_request__(self):  
        print('loading sample element')
        sample_frame = np.random.random((1080, 1920, 3))
        try:
            for i in range(self.buffer_shape):
                self.predict(sample_frame)
        except Exception as e:
            print(e)
        for i in range(self.buffer_shape):
            self.postprocess_op()
        self.cursor_id = 0
        self.frame_processed = 0






    def preprocess_frame(self,frame):
        """
        after processing cursor id is updated
        """
        in_frame = cv2.resize(frame.copy(), (self.width, self.height))
        in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        in_frame = in_frame.reshape((self.batch_size, self.channel, self.height, self.width))
        self.in_frame = in_frame
    

    def predict_sync(self,frame):
        """
        predict sync only
        return output_values
        """
        self.preprocess_frame(frame)
        self.exec_net.requests[0].wait()
        self.exec_net.requests[0].infer({self.input_blob: self.in_frame})
        return [self.exec_net.requests[0].outputs[node] for node in self.out_blob ]      
        #         if self.start_infer :
#             if self.net.requests[cur_request_id].wait(-1) == 0 :
#                 self.output = [exec_net.requests[cur_request_id].outputs[node] for node in self.output_blob]
#         else :Sampad Mahapatra
#             if len(self.frames_buffer)>=self.buffer_shape:
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
            self.cursor_id = (self.cursor_id+1)% self.buffer_shape

            self.frame_processed+=1
        except Exception as e:
            print("withing preprocess_frame "+str(self.model_xml.split("/")[-1]),e)

        #         if self.start_infer :
#             if self.net.requests[cur_request_id].wait(-1) == 0 :
#                 self.output = [exec_net.requests[cur_request_id].outputs[node] for node in self.output_blob]
#         else :
#             if len(self.frames_buffer)>=self.buffer_shape:
#                 self.start_infer = True
                
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
        self.output = [self.exec_net.requests[request_id].outputs[node] for node in self.out_blob]
        op_frame = self.frames_buffer[request_id]
        attr = self.attrs[request_id]

        return op_frame,attr,self.output

            
            
