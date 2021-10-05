#!/usr/bin/env python
# coding: utf-8

# In[1]:


detection_model_path = '/home/prateek/prateek_space/model_files/openvino_model/Retinanet/2019_06_20_11_59_58_resnet50_csv_100.xml'
cpu_extension = '../build_samples/intel64/Release/lib/libcpu_extension.so'
device = 'CPU'


# In[2]:


import time,cv2
import numpy as np
import IPython.display as Disp
import support_utility_openvino


# In[3]:


retinanet_detection = support_utility_openvino.async_infer(3)
retinanet_detection.load_model(cpu_exension_path=cpu_extension,model_path=detection_model_path,device=device)
# try:
#     retinanet_detection.predict(np.random.random((1080, 1920, 3)))
# except Exception as e:
#     print(e)
# retinanet_detection.cursor_id = 1
# retinanet_detection.frame_processed = 0
output_support = support_utility_openvino.create_plot()


# In[4]:


labels = ['person','helmet','no_helmet','vest','no_vest','worker']
bbox_colors = [(0,0,0),(0,255,0),(0,0,255),(0,255,0),(0,0,255),(255,0,0)]


# In[5]:


cap = cv2.VideoCapture('/media/prateek/prateek_space/dataset/helmet_vest_violation_data/Camera6_spandan office_spandan office_20181219030822_20181219030835_2990866.mp4')
cv2.namedWindow("preview",cv2.WINDOW_NORMAL)
ret,frame = cap.read()
fps = []
thresh = .5
# dets = []
while True:
    fps = fps[-100:]
#     Disp.clear_output(wait=True)
    ret,frame = cap.read()
    if not ret:
        break
    t1 = time.time()
    retinanet_detection.predict(frame)
    if retinanet_detection.frame_processed>retinanet_detection.buffer_shape:
        try:
            frame,attr,res = retinanet_detection.postprocess_op()
            res[0][:,:,:,1] = res[0][:,:,:,1]+1
#             op,bboxes = output_support.trim_frame_with_result(frame,res[0],threshold=thresh)
#             attrs = [attr_detection.predict_sync(vehicle) for vehicle in op]
#             attrs = [(vehicle_colors[np.argmax(att[0])], vehicle_type[np.argmax(att[1])]) for att in attrs]
#             dets.append(len(attrs))
            fps.append(1/(time.time()-t1))
            op_frame = output_support.write_text(frame,"FPS = %.2f"%np.mean(fps),text_color = (0,0,0),font_scale=1,font_thickness=2,highlight_color = (127,0,255))
#             for idx,b in enumerate(bboxes):
#                 op_frame = output_support.create_bbox_with_text(op_frame,(tuple(b[:2]),tuple(b[2:])),",".join(attrs[idx]))
            op_frame = output_support.label_obj_detection(op_frame,res[0],labels=labels,colors=bbox_colors,font_scale=1,font_thickness = 1,thresh=.5)
        except Exception as e:
            print(e)
#         print(retinanet_detection.cursor_id,retinanet_detection.frame_processed)
        cv2.imshow("preview",op_frame)
        k = cv2.waitKey(1)
        if k==27:
            break
cap.release()
cv2.destroyAllWindows()
        
        
    


# In[6]:


retinanet_detection.cursor_id


# In[ ]:


retinanet_detection.__load_first_request__()


# In[ ]:




