#!/usr/bin/env python
"""
 Copyright (c) 2018 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import logging as log
import os.path as osp
import sys,pdb,math
import time
from datetime import datetime
from argparse import ArgumentParser
import cv2
import matplotlib as mpl

mpl.rcParams['ytick.labelsize'] = 'xx-large' 
mpl.rcParams['font.weight'] = 'bold'
mpl.rcParams['figure.autolayout'] = True
mpl.rcParams['xtick.bottom'] = False
import matplotlib.pyplot as plt
plt.style.use('dark_background')

import numpy as np

from openvino.inference_engine import IENetwork
from ie_module import InferenceContext
from landmarks_detector import LandmarksDetector
from face_detector import FaceDetector
from faces_database import FacesDatabase
from face_identifier import FaceIdentifier
import pytz
india_tz = pytz.timezone('Asia/Kolkata')
from elk import Elk

DEVICE_KINDS = ['CPU', 'GPU', 'FPGA', 'MYRIAD', 'HETERO', 'HDDL']

sys.path.append('..')
import support_utility_openvino

def build_argparser():
    parser = ArgumentParser()

    general = parser.add_argument_group('General')
    general.add_argument('-i', '--input', metavar="PATH", default='0',
                         help="(optional) Path to the input video " \
                         "('0' for the camera, default)")
    general.add_argument('-o', '--output', metavar="PATH", default="",
                         help="(optional) Path to save the output video to")
    general.add_argument('--no_show', action='store_true',
                         help="(optional) Do not display output")
    general.add_argument('-tl', '--timelapse', action='store_true',
                         help="(optional) Auto-pause after each frame")
    general.add_argument('-cw', '--crop_width', default=0, type=int,
                         help="(optional) Crop the input stream to this width " \
                         "(default: no crop). Both -cw and -ch parameters " \
                         "should be specified to use crop.")
    general.add_argument('-ch', '--crop_height', default=0, type=int,
                         help="(optional) Crop the input stream to this height " \
                         "(default: no crop). Both -cw and -ch parameters " \
                         "should be specified to use crop.")

    gallery = parser.add_argument_group('Faces database')
    gallery.add_argument('-fg', metavar="PATH", required=True,
                         help="Path to the face images directory")
    gallery.add_argument('--run_detector', action='store_true',
                         help="(optional) Use Face Detection model to find faces" \
                         " on the face images, otherwise use full images.")

    models = parser.add_argument_group('Models')
    models.add_argument('-m_fd', metavar="PATH", default="", required=True,
                        help="Path to the Face Detection model XML file")
    models.add_argument('-m_lm', metavar="PATH", default="", required=True,
                        help="Path to the Facial Landmarks Regression model XML file")
    models.add_argument('-m_reid', metavar="PATH", default="", required=True,
                        help="Path to the Face Reidentification model XML file")
    models.add_argument('-m_ag', metavar="PATH", default="", required=False,
                        help="Path to the Age Gender model XML file")
    models.add_argument('-m_em', metavar="PATH", default="", required=False,
                        help="Path to the Emotions model XML file")
    #--------------------------------------------------------------#
    models.add_argument('--use_ag_em', action='store_true',
                             help="use age gender emotion model")
    #----------------------------------------------------------------#

    infer = parser.add_argument_group('Inference options')
    infer.add_argument('-d_fd', default='CPU', choices=DEVICE_KINDS,
                       help="(optional) Target device for the " \
                       "Face Detection model (default: %(default)s)")
    infer.add_argument('-d_lm', default='CPU', choices=DEVICE_KINDS,
                       help="(optional) Target device for the " \
                       "Facial Landmarks Regression model (default: %(default)s)")
    infer.add_argument('-d_reid', default='CPU', choices=DEVICE_KINDS,
                       help="(optional) Target device for the " \
                       "Face Reidentification model (default: %(default)s)")
    infer.add_argument('-d_ag', default='CPU', choices=DEVICE_KINDS,
                       help="(optional) Target device for the " \
                       "Age Gender model (default: %(default)s)")
    infer.add_argument('-d_em', default='CPU', choices=DEVICE_KINDS,
                       help="(optional) Target device for the " \
                       "emotion model (default: %(default)s)")

    infer.add_argument('-l', '--cpu_lib', metavar="PATH", default="",
                       help="(optional) For MKLDNN (CPU)-targeted custom layers, if any. " \
                       "Path to a shared library with custom layers implementations")
    infer.add_argument('-c', '--gpu_lib', metavar="PATH", default="",
                       help="(optional) For clDNN (GPU)-targeted custom layers, if any. " \
                       "Path to the XML file with descriptions of the kernels")
    infer.add_argument('-v', '--verbose', action='store_true',
                       help="(optional) Be more verbose")
    infer.add_argument('-pc', '--perf_stats', action='store_true',
                       help="(optional) Output detailed per-layer performance stats")
    infer.add_argument('-t_fd', metavar='[0..1]', type=float, default=0.6,
                       help="(optional) Probability threshold for face detections" \
                       "(default: %(default)s)")

    infer.add_argument('-t_id', metavar='[0..1]', type=float, default=0.3,
                       help="(optional) Cosine distance threshold between two vectors " \
                       "for face identification (default: %(default)s)")

    #-------------------------------------------------------------------------------
    infer.add_argument('-min_resolution', metavar='[1000-40000]', type=int, default=1000,
                       help="minimum resolution to use for face detection" \
                       "(default: %(default)s)")
    infer.add_argument('-relative_size', metavar='[0..1]', type=float, default=.7,
                       help="scale parameter by which plot will be resized as per face detected" \
                       "(default: %(default)s)")
    infer.add_argument('-alpha', metavar='[0..1]', type=float, default=.4,
                       help="opacity for plot to be drawn" \
                       "(default: %(default)s)")
    infer.add_argument('--data_push', action='store_true',
                             help="push data to database")
    #-------------------------------------------------------------------------------

    infer.add_argument('-exp_r_fd', metavar='NUMBER', type=float, default=1.15,
                       help="(optional) Scaling ratio for bboxes passed to face recognition " \
                       "(default: %(default)s)")
    infer.add_argument('--allow_grow', action='store_true',
                       help="(optional) Allow to grow faces gallery and to dump on disk. " \
                       "Available only if --no_show option is off.")

    return parser


# def load_ag_em(self,model_path,device,cpu_extension=None,gpu_extension=None):
#   plugin = IEPlugin(device=device)
#   if device=='CPU' and cpu_extension is not None:
#     plugin.add_cpu_extension(cpu_extension)
#   if device=='GPU' and gpu_extension is not None:
#     plugin.set_config({"CONFIG_FILE": gpu_extension})
#   model = IENetwork(model=m_fd, weights=os.path.splitext(m_fd)[0] + ".bin")


class FrameProcessor:
    QUEUE_SIZE = 16

    def __init__(self, args):
        used_devices = set([args.d_fd, args.d_lm, args.d_reid])
        self.context = InferenceContext()
        context = self.context
        context.load_plugins(used_devices, args.cpu_lib, args.gpu_lib)
        for d in used_devices:
            context.get_plugin(d).set_config({
                "PERF_COUNT": "YES" if args.perf_stats else "NO"})

        log.info("Loading models")
        face_detector_net = self.load_model(args.m_fd)
        landmarks_net = self.load_model(args.m_lm)
        face_reid_net = self.load_model(args.m_reid)

        self.face_detector = FaceDetector(face_detector_net,
                                          confidence_threshold=args.t_fd,
                                          roi_scale_factor=args.exp_r_fd)
        self.landmarks_detector = LandmarksDetector(landmarks_net)
        self.face_identifier = FaceIdentifier(face_reid_net,
                                              match_threshold=args.t_id)
        ##--------------------prateek code-----------------------------------------------------------------------------------
        self.detect_ag_em = args.use_ag_em
        self.min_res = args.min_resolution

        if self.detect_ag_em:
          try:
            self.em_detector = support_utility_openvino.async_infer()
            self.em_detector.load_model(args.m_em,args.d_em,cpu_exension_path=args.cpu_lib,gpu_extension_path=args.gpu_lib)
            self.ag_detector = support_utility_openvino.async_infer()
            self.ag_detector.load_model(args.m_ag,args.d_ag,cpu_exension_path=args.cpu_lib,gpu_extension_path=args.gpu_lib)
          except:
            log.error("Cannot load model: %s" % args.d_ag)
            log.error("Cannot load model: %s" % args.d_em)
            self.detect_ag_em = False
            args.use_ag_em = False



        #--------------------------------------------------------------------------------------------------------------------





        self.face_detector.deploy(args.d_fd, context)
        self.landmarks_detector.deploy(args.d_lm, context,
                                       queue_size=self.QUEUE_SIZE)
        self.face_identifier.deploy(args.d_reid, context,
                                    queue_size=self.QUEUE_SIZE)
        log.info("Models are loaded")

        log.info("Building faces database using images from '%s'" % (args.fg))
        self.faces_database = FacesDatabase(args.fg, self.face_identifier,
                                            self.landmarks_detector,
                                            self.face_detector if args.run_detector else None, args.no_show)
        self.face_identifier.set_faces_database(self.faces_database)
        log.info("Database is built, registered %s identities" % \
            (len(self.faces_database)))

        self.allow_grow = args.allow_grow and not args.no_show

    

    def load_model(self, model_path):
        model_path = osp.abspath(model_path)
        model_description_path = model_path
        model_weights_path = osp.splitext(model_path)[0] + ".bin"
        log.info("Loading the model from '%s'" % (model_description_path))
        assert osp.isfile(model_description_path), \
            "Model description is not found at '%s'" % (model_description_path)
        assert osp.isfile(model_weights_path), \
            "Model weights are not found at '%s'" % (model_weights_path)
        model = IENetwork(model_description_path, model_weights_path)
        log.info("Model is loaded")
        return model

    def process(self, frame):
        assert len(frame.shape) == 3, \
            "Expected input frame in (H, W, C) format"
        assert frame.shape[2] in [3, 4], \
            "Expected BGR or BGRA input"

        orig_image = frame.copy()
        frame = frame.transpose((2, 0, 1)) # HWC to CHW
        frame = np.expand_dims(frame, axis=0)

        self.face_detector.clear()
        self.landmarks_detector.clear()
        self.face_identifier.clear()

        self.face_detector.start_async(frame)
        rois = self.face_detector.get_roi_proposals(frame)
        if self.QUEUE_SIZE < len(rois):
            log.warning("Too many faces for processing." \
                    " Will be processed only %s of %s." % \
                    (self.QUEUE_SIZE, len(rois)))
            rois = rois[:self.QUEUE_SIZE]

        op_rois = []
        for idx,roi in enumerate(rois):
          if roi.size.prod()>=self.min_res:
            op_rois.append(roi)
        rois = op_rois
        del op_rois

        self.landmarks_detector.start_async(frame, rois)
        landmarks = self.landmarks_detector.get_landmarks()
        # pdb.set_trace()
        if self.detect_ag_em:
          # x1,y1,x2,y2 = int(face.position[0]),int(face.position[1]),int(face.position[0]+face.size[0]),int(face.position[1]+face.size[1])
          faces = [orig_image[int(face.position[1]):int(face.position[1]+face.size[1]),int(face.position[0]):int(face.position[0]+face.size[0])] for face in rois]
          ems = [self.em_detector.predict_sync(face) for face in faces]
          ags = [self.ag_detector.predict_sync(face) for face in faces]

        self.face_identifier.start_async(frame, rois, landmarks)
        face_identities, unknowns = self.face_identifier.get_matches()
        if self.allow_grow and len(unknowns) > 0:
            for i in unknowns:
                # This check is preventing asking to save half-images in the boundary of images
                if rois[i].position[0] == 0.0 or rois[i].position[1] == 0.0 or \
                    (rois[i].position[0] + rois[i].size[0] > orig_image.shape[1]) or \
                    (rois[i].position[1] + rois[i].size[1] > orig_image.shape[0]):
                    continue
                crop = orig_image[int(rois[i].position[1]):int(rois[i].position[1]+rois[i].size[1]), int(rois[i].position[0]):int(rois[i].position[0]+rois[i].size[0])]
                name = self.faces_database.ask_to_save(crop)
                if name:
                    id = self.faces_database.dump_faces(crop, face_identities[i].descriptor, name)
                    face_identities[i].id = id

        outputs = [rois, landmarks, face_identities]
        if self.detect_ag_em:
          outputs = outputs+[ems,ags]


        return outputs


    def get_performance_stats(self):
        stats = {
            'face_detector': self.face_detector.get_performance_stats(),
            'landmarks': self.landmarks_detector.get_performance_stats(),
            'face_identifier': self.face_identifier.get_performance_stats(),
        }
        return stats


class Visualizer:
    BREAK_KEY_LABELS = "q(Q) or Escape"
    BREAK_KEYS = {ord('q'), ord('Q')}


    def __init__(self, args):
        self.frame_processor = FrameProcessor(args)
        self.display = not args.no_show
        self.print_perf_stats = args.perf_stats

        self.frame_time = 0
        self.frame_start_time = 0
        self.fps = 0
        self.frame_num = 0
        self.frame_count = -1
        self.detect_ag_em = args.use_ag_em
        self.figure = plt.figure(figsize=(8, 6))
        self.input_crop = None
        self.alpha =args.alpha
        self.relative_size = args.relative_size
        self.male_im = cv2.imread('male.jpg')
        self.male_color = (180,81,23)
        self.female_color = (161,94,223)
        self.female_im = cv2.imread('female.jpg')
        self.data_push = None
        if args.data_push:
            self.data_push = Elk()

        if args.crop_width and args.crop_height:
            self.input_crop = np.array((args.crop_width, args.crop_height))

        self.frame_timeout = 0 if args.timelapse else 1

    def update_fps(self):
        now = time.time()
        self.frame_time = now - self.frame_start_time
        self.fps = 1.0 / self.frame_time
        self.frame_start_time = now

    def draw_text_with_background(self, frame, text, origin,
                                  font=cv2.FONT_HERSHEY_SIMPLEX, scale=1.0,
                                  color=(0, 0, 0), thickness=1, bgcolor=(255, 255, 255)):
        text_size, baseline = cv2.getTextSize(text, font, scale, thickness)
        cv2.rectangle(frame,
                      tuple((origin + (0, baseline)).astype(int)),
                      tuple((origin + (text_size[0], -text_size[1])).astype(int)),
                      bgcolor, cv2.FILLED)
        cv2.putText(frame, text,
                    tuple(origin.astype(int)),
                    font, scale, color, thickness)
        return text_size, baseline
    def fig2data (self,fig):
        """
        @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
        @param fig a matplotlib figure
        @return a numpy 3D array of RGBA values
        """
        # draw the renderer
        fig.canvas.draw ()
     
        # Get the RGBA buffer from the figure
        w,h = fig.canvas.get_width_height()
    #     w, h = fig.get_size_inches() * fig.get_dpi()
        buf = np.fromstring (fig.canvas.tostring_rgb(), dtype=np.uint8 ).reshape(int(h), int(w), 3)
        #buf.shape = ( w, h,3 )
     
        # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
        #buf = np.roll ( buf, 3, axis = 2 )
        return buf
    def overlay_im(self,src_im,overlay_image,location,size,alpha):
        """
        src_im : base image(larger image)
        overlay_image : image to be overlayed
        location : bottom left point of overlay image
        size : size of image to be overlayed
        alpha : opacity of the overlayed image
        """
        try:
          orig_img = src_im.copy()

          resized_im = cv2.resize(overlay_image,(min(orig_img.shape[1]-location[0],size[0]),min(orig_img.shape[0]-location[1],size[1])))
          #     print(resized_im.shape)
          orig_img[location[1]:location[1]+resized_im.shape[0], location[0]:location[0]+resized_im.shape[1]] = resized_im
          #     src_im[white_im!=255] = white_im
          src_im[:,:,:] =  cv2.addWeighted(orig_img, (1-alpha),src_im , alpha, 0.0)
        except Exception as e:
          log.error("Cannot create image overlay: %s" % e)
        # pdb.set_trace()

    def draw_detection_roi(self, frame, roi, identity,ags=None,ems=None):
        label = self.frame_processor \
            .face_identifier.get_identity_label(identity.id)
        if ags is not None:
          color = self.female_color
          if ags[1].argmax()==1:#male
            color = self.male_color
        else:
          color = (0,255,0)
        # Draw face ROI border
        cv2.rectangle(frame,
                      tuple(roi.position), tuple(roi.position + roi.size),
                      color, 2)

        # Draw identity label
        text_scale = .8
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize("H1", font, text_scale, 1)
        line_height = np.array([0, text_size[0][1]])
        text = label

        #-----------------------original-----------------------------
        if ems is  None:
          if identity.id != FaceIdentifier.UNKNOWN_ID:
              text += ' %.2f%%' % (100.0 * (1 - identity.distance))
        #-----------------------changed-----------------------------
        else:
          if identity.id != FaceIdentifier.UNKNOWN_ID:
              text += '  (Happiness Score=%.2f%%)' % (100.0 * ems[0].flatten()[1])       
        #-----------------------------------------------------------
        self.draw_text_with_background(frame, text,
                                       roi.position - line_height * 0.5,
                                       font, scale=text_scale,thickness=2)

    def draw_detection_keypoints(self, frame, roi, landmarks):
        keypoints = [landmarks.left_eye,
                     landmarks.right_eye,
                     landmarks.nose_tip,
                     landmarks.left_lip_corner,
                     landmarks.right_lip_corner]

        for point in keypoints:
            center = roi.position + roi.size * point
            cv2.circle(frame, tuple(center.astype(int)), 2, (0, 255, 255), 2)


    def draw_detection_emotions(self,frame,roi,ems,relative_size = .7,alpha=.4,write_values = False):
      try:
        p1 = roi.position.astype(int)
        p2 = p1+roi.size.astype(int)
        ems = ems[0]
        self.figure.clear()
        # figure.patch.set_facecolor('blue')
        # figure.patch.set_alpha(0)
        ax_1   = self.figure.add_subplot(111)
        # ax_1.set_facecolor('black')
        bar = ax_1.barh([ 'neutral', 'happy', 'sad', 'surprise', 'anger'],ems.flatten())
        if write_values:
          for rect in bar:
            x = rect.get_x()
            y = rect.get_y()
            width = rect.get_width()
            height = rect.get_height()
            ax_1.text( rect.get_height(),y+width/2, '%d' % int(100*width), ha='center', va='bottom', fontsize=20)
        # ax_1.set_axis_off()
        #     df = pd.DataFrame([np.random.random(),np.random.random(),np.random.random(),np.random.random()],\
        #                        index=['happy','anger','sad','disgust'],columns=['probs'])
        #     df.plot(kind='barh',ax = ax_1,figsize=(16,12),fontsize=40)

        #     figure.patch.set_visible(False)

        #     ax_1.axis('off')
        buf = self.fig2data(self.figure)
        # buf = cv2.resize(buf,(frame.shape[1],frame.shape[0]))
        buf = cv2.cvtColor(buf,cv2.COLOR_RGB2BGR)

        #     frame_1 = cv2.addWeighted(frame, alpha, buf, beta, 0.0)
        #     frame_1 = np.concatenate((frame,cv2.resize(buf, (2*frame.shape[1],frame.shape[0]))),axis=1)
        self.overlay_im(frame,buf,(p2[0],int(p1[1])),((1,self.relative_size)*roi.size).astype(int),self.alpha)
      except Exception as e:
        log.error("Cannot draw Emotions %s" % e)



    def draw_detection_age_gender(self,frame,roi,ags):
      """
      output : age/100
      prob : [female, male]
      """
      try:
        color = (0, 0, 0)
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_scale = 1      
        # pdb.set_trace()
        age = ags[0].flatten()[0]*10
        age_band = str(math.floor(age)*10)+"-"+str(math.ceil(age)*10)
        thickness = 2
        text_size = cv2.getTextSize(age_band, font, text_scale, thickness)
        origin = np.array([frame.shape[-2] - text_size[0][0] - 10, 10])
        if ags[1].argmax()==1:#male
          # male_pic = cv2.resize(self.male_im,.5*self.relative_size*roi.size)
          x1,y1 = int(roi.position[0]+roi.size[0]),int(roi.position[1]+self.relative_size*roi.size[1])
          self.overlay_im(frame,self.male_im,(x1,y1),((1-self.relative_size)*roi.size).astype(int),self.alpha)
          origin = np.array((x1+int((1-self.relative_size)*roi.size[0]),y1+int(text_size[0][1])))
          self.draw_text_with_background(frame,age_band,origin,color =(255,255,255),scale=text_scale,thickness=thickness,bgcolor = self.male_color)
          # cv2.putText(frame, age_band,origin, font, text_scale, self.male_color, thickness)
        else:
          # female_pic = cv2.resize(self.female_im,.5*self.relative_size*roi.size)
          x1,y1 = int(roi.position[0]+roi.size[0]),int(roi.position[1]+self.relative_size*roi.size[1])
          self.overlay_im(frame,self.female_im,(x1,y1),((1-self.relative_size)*roi.size).astype(int),self.alpha)
          origin = np.array((x1+int((1-self.relative_size)*roi.size[0]),y1+int(text_size[0][1])))
          self.draw_text_with_background(frame,age_band,origin,color =(255,255,255),scale=text_scale,thickness=thickness,bgcolor = self.female_color)
      except Exception as e:
          log.error("Cannot draw age gender %s" % e)



    def draw_detections(self, frame, detections):
        if  self.detect_ag_em:
            for roi, landmarks, identity,ems,ags, in zip(*detections):
            
                self.draw_detection_roi(frame, roi, identity,ags,ems)
                # self.draw_detection_keypoints(frame, roi, landmarks)
                self.draw_detection_emotions(frame, roi, ems)
                self.draw_detection_age_gender(frame, roi, ags)
                if self.data_push is not None:
                    push_data = {}
                    person_id = self.frame_processor.face_identifier.get_identity_label(identity.id)
                    if person_id=="Unknown":
                      break
                    push_data['identity'] = person_id.replace("_"," ")
                    push_data['timeStamp'] = datetime.now(tz=india_tz).isoformat()#.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
                    #YYYY-MM-DD'T'HH:mm:ssZ "2016-07-15T15:29:50+02:00" elk format
                    push_data['age'] = int(ags[0].flatten()[0]*100)
                    if ags[1].argmax()==1:
                        push_data['gender'] = 'male'
                    else:
                        push_data['gender'] = 'female'
                    ems_flatten = ems[0].flatten()
                    for idx,em in enumerate([ 'neutral', 'happy', 'sad', 'surprise', 'anger']):
                        push_data["emotions_"+em] = int(ems_flatten[idx]*100)
                    del ems_flatten
                    self.data_push.push_data(push_data)
      
        else:
            for roi, landmarks, identity, in zip(*detections):
                self.draw_detection_roi(frame, roi, identity)
                # self.draw_detection_keypoints(frame, roi, landmarks)



    def draw_status(self, frame, detections):
        origin = np.array([10, 10])
        color = (10, 160, 10)
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_scale = 0.5
        text_size, _ = self.draw_text_with_background(frame,
                                                      "Frame time: %.3fs" % (self.frame_time),
                                                      origin, font, text_scale, color)
        self.draw_text_with_background(frame,
                                       "FPS: %.1f" % (self.fps),
                                       (origin + (0, text_size[1] * 1.5)), font, text_scale, color)

        log.debug('Frame: %s/%s, detections: %s, ' \
                  'frame time: %.3fs, fps: %.1f' % \
                     (self.frame_num, self.frame_count, len(detections[-1]), self.frame_time, self.fps))

        if self.print_perf_stats:
            log.info('Performance stats:')
            log.info(self.frame_processor.get_performance_stats())

    def display_interactive_window(self, frame):
        color = (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_scale = 0.5
        text = "Press '%s' key to exit" % (self.BREAK_KEY_LABELS)
        thickness = 2
        text_size = cv2.getTextSize(text, font, text_scale, thickness)
        origin = np.array([frame.shape[-2] - text_size[0][0] - 10, 10])
        line_height = np.array([0, text_size[0][1]]) * 1.5
        cv2.putText(frame, text,
                    tuple(origin.astype(int)), font, text_scale, color, thickness)

        # cv2.namedWindow('Face recognition demo', cv2.WINDOW_NORMAL)
        cv2.imshow('Happiness Meter ', frame)

    def should_stop_display(self):
        key = cv2.waitKey(self.frame_timeout) & 0xFF
        return key in self.BREAK_KEYS


    def process(self, input_stream, output_stream):
        self.input_stream = input_stream
        self.output_stream = output_stream
        cv2.namedWindow('Happiness Meter ',cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Happiness Meter ', 640, 480)
        while input_stream.isOpened():
            has_frame, frame = input_stream.read()
            # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            if not has_frame:
                break

            if self.input_crop is not None:
                frame = Visualizer.center_crop(frame, self.input_crop)
            detections = self.frame_processor.process(frame)
            

            # pdb.set_trace()
            self.draw_detections(frame, detections)
            self.draw_status(frame, detections)

            if output_stream:
                output_stream.write(frame)
            if self.display:
                self.display_interactive_window(frame)
                if self.should_stop_display():
                    break

            self.update_fps()
            self.frame_num += 1

    @staticmethod
    def center_crop(frame, crop_size):
        fh, fw, fc = frame.shape
        crop_size[0] = min(fw, crop_size[0])
        crop_size[1] = min(fh, crop_size[1])
        return frame[(fh - crop_size[1]) // 2 : (fh + crop_size[1]) // 2,
                     (fw - crop_size[0]) // 2 : (fw + crop_size[0]) // 2,
                     :]
    def make_1080p(self,cap):

        cap.set(3, 1920)
        cap.set(4, 1080)
        cap.set(5,30)
    def run(self, args):
        input_stream = Visualizer.open_input_stream(args.input)
        try:
          self.make_1080p(input_stream)
        except Exception as e:
          log.error("Cannot resize input stream: %s" % e)
        if input_stream is None or not input_stream.isOpened():
            log.error("Cannot open input returnstream: %s" % args.input)
        fps = input_stream.get(cv2.CAP_PROP_FPS)
        frame_size = (int(input_stream.get(cv2.CAP_PROP_FRAME_WIDTH)),
                      int(input_stream.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.frame_count = int(input_stream.get(cv2.CAP_PROP_FRAME_COUNT))
        if args.crop_width and args.crop_height:
            crop_size = (args.crop_width, args.crop_height)
            frame_size = tuple(np.minimum(frame_size, crop_size))
        log.info("Input stream info: %d x %d @ %.2f FPS" % \
            (frame_size[0], frame_size[1], fps))
        output_stream = Visualizer.open_output_stream(args.output, fps, frame_size)

        self.process(input_stream, output_stream)

        # Release resources
        if output_stream:
            output_stream.release()
        if input_stream:
            input_stream.release()

        cv2.destroyAllWindows()


    @staticmethod
    def open_input_stream(path):
        log.info("Reading input data from '%s'" % (path))
        stream = path
        try:
            stream = int(path)
        except ValueError:
            pass
        return cv2.VideoCapture(stream)

    @staticmethod
    def open_output_stream(path, fps, frame_size):
        output_stream = None
        if path != "":
            if not path.endswith('.avi'):
                log.warning("Output file extension is not 'avi'. " \
                        "Some issues with output can occur, check logs.")
            log.info("Writing output to '%s'" % (path))
            output_stream = cv2.VideoWriter(path,
                                            cv2.VideoWriter.fourcc(*'MJPG'), fps, frame_size)
        return output_stream


def main():
    args = build_argparser().parse_args()

    log.basicConfig(format="[ %(levelname)s ] %(asctime)-15s %(message)s",
                    level=log.INFO if not args.verbose else log.DEBUG, stream=sys.stdout)

    log.debug(str(args))

    visualizer = Visualizer(args)
    visualizer.run(args)


if __name__ == '__main__':
    main()
