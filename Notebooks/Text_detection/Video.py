import cv2 as cv
import numpy as np
import pytesseract
from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import imutils
import time

from imutils.object_detection import non_max_suppression
import os
from openvino.inference_engine import IENetwork, IEPlugin

def decode_predictions(scores, geometry):
    # grab the number of rows and columns from the scores volume, then
    # initialize our set of bounding box rectangles and corresponding
    # confidence scores
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []
    
    # loop over the number of rows
    for y in range(0, numRows):
        # extract the scores (probabilities), followed by the
        # geometrical data used to derive potential bounding box
        # coordinates that surround text
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]
        
        # loop over the number of columns
        for x in range(0, numCols):
            # if our score does not have sufficient probability,
            # ignore it
            if scoresData[x] < 0.5:
                continue
        
            # compute the offset factor as our resulting feature
            # maps will be 4x smaller than the input image
            (offsetX, offsetY) = (x * 4.0, y * 4.0)
            
            # extract the rotation angle for the prediction and
            # then compute the sin and cosine
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)
            
            # use the geometry volume to derive the width and height
            # of the bounding box
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]
            
            # compute both the starting and ending (x, y)-coordinates
            # for the text prediction bounding box
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)
            
            # add the bounding box coordinates and probability score
            # to our respective lists
            rects.append((startX, startY, endX, endY))
        confidences.append(scoresData[x])

# return a tuple of the bounding boxes and associated confidences
    return (rects, confidences)

model_bin="/Users/20widyatmodjok/openvino_models/ir/FP32/classification/squeezenet/1.1/caffe/squeezenet1.1.bin"
model_xml="/Users/20widyatmodjok/openvino_models/ir/FP32/classification/squeezenet/1.1/caffe/squeezenet1.1.xml"

model_bin = "frozen_east_text_detection.bin"
model_xml = "frozen_east_text_detection.xml"

plugin =IEPlugin(device="CPU")
plugin.add_cpu_extension("/Users/20widyatmodjok/inference_engine_samples_build/intel64/Release/lib/libcpu_extension.dylib")

net = IENetwork(model=model_xml, weights=model_bin)

exec_net = plugin.load(network=net)


cap = cv.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    
    orig = frame.copy()
    (origH, origW) = frame.shape[:2]
    (newW, newH) = (320, 320)
    rW = origW / float(newW)
    rH = origH / float(newH)

    frame = cv.dnn.blobFromImage(frame,1,(320,320),(512,512))#1,(320,320),(512,512
    #img = img.transpose((1,0,2))
    #img = img.shape

    #frame = frame.astype('uint8')
    img2=frame
    #img2 = img.shape(img2, (2,2))


    #img.shape

    #cv.imshow("test", img)
    #cv.waitKey(0)


    input_blob = next(iter(net.inputs))
    #cap = cv.VideoCapture(0)


    #np.ndarray(shape=(2,4), dtype=float, order='F')


    output_blob = next(iter(net.outputs))
    output = exec_net.infer(inputs={input_blob:frame})


    scores = output["feature_fusion/Conv_7/Sigmoid"]
    geometry = output["feature_fusion/concat_3"]

    print(scores)
    print(geometry)
    (rects, confidences) = decode_predictions(scores, geometry)


    boxes = non_max_suppression(np.array(rects), probs=confidences)

    print(boxes)
    # initialize the list of results
    results = []

    # loop over the bounding boxes
    for (startX, startY, endX, endY) in boxes:
        # scale the bounding box coordinates based on the respective
        # ratios
        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)
        
        # in order to obtain a better OCR of the text we can potentially
        # apply a bit of padding surrounding the bounding box -- here we
        # are computing the deltas in both the x and y directions
        #dX = int((endX - startX) * args["padding"])
        #dY = int((endY - startY) * args["padding"])
        
        dX = 0
        dY = 0
        # apply padding to each side of the bounding box, respectively
        startX = max(0, startX - dX)
        startY = max(0, startY - dY)
        endX = min(origW, endX + (dX * 2))
        endY = min(origH, endY + (dY * 2))
        
        # extract the actual padded ROI
        roi = orig[startY:endY, startX:endX]
        
        # in order to apply Tesseract v4 to OCR text we must supply
        # (1) a language, (2) an OEM flag of 4, indicating that the we
        # wish to use the LSTM neural net model for OCR, and finally
        # (3) an OEM value, in this case, 7 which implies that we are
        # treating the ROI as a single line of text
        config = ("-l eng --oem 1 --psm 7")
        text = pytesseract.image_to_string(roi, config=config)
        
        # add the bounding box coordinates and OCR'd text to the list
        # of results
        results.append(((startX, startY, endX, endY), text))

    # sort the results bounding box coordinates from top to bottom
    results = sorted(results, key=lambda r:r[0][1])

    # loop over the results
    for ((startX, startY, endX, endY), text) in results:
        # display the text OCR'd by Tesseract
        print("OCR TEXT")
        print("========")
        print("{}\n".format(text))
        
        # strip out non-ASCII text so we can draw the text on the image
        # using OpenCV, then draw the text and a bounding box surrounding
        # the text region of the input image
        text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
        output = orig.copy()
        cv.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
        cv.putText(frame, text, (startX, startY - 20), cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        #gray = cv.cvtColor(outputs,cv.COLOR_BGR2GRAY)
    
    
    
        cv.imshow('frame', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
        # show the output image
        cv.imshow("Text Detection", frame)
        cv.waitKey(0)

