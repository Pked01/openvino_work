#!/usr/bin/env python
'''
 Copyright (C) 2018-2019 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
'''
import dlib,pdb
import glob
import os
import json
import sys

if len(sys.argv) == 2:
    dir = sys.argv[1]
else:
    dir = '.' + os.sep

files_list = glob.glob(dir + '*.png') + glob.glob(dir + '*.jpg') + glob.glob(dir + '*.jpeg') +glob.glob(dir + '*.bmp')
detector = dlib.get_frontal_face_detector()

labels = []
objects = {}
os.makedirs(os.path.join(dir, 'cropped_image'),exist_ok=True)
for file in files_list:
    try:
        # pdb.set_trace()
        image = dlib.load_rgb_image(file)
        dets = detector(image,1)[0]
        # face_locations = face_recognition.face_locations(image)
        image_1  =  image[dets.top():dets.bottom(),dets.left():dets.right(),:]
        image_1 = dlib.resize_image(image_1,300,300)
        file_name = os.path.basename(file)
        file_name = file_name.replace(".","-",1)
        dlib.save_image(image_1,'cropped_image/'+file_name)
    except Exception as e:
        print(e)
        print('increasing upsample')
        ## save original image 
        dlib.save_image(image,'cropped_image/'+file_name)
        for i in range(5):
            try: 
                dets = detector(image,1)[0]
                # face_locations = face_recognition.face_locations(image)
                image_1  =  image[dets.top():dets.bottom(),dets.left():dets.right(),:]
                file_name = os.path.basename(file)
                dlib.save_image(image_1,'cropped_image/'+file_name)
                break
            except Exception as e:
                print(e)


