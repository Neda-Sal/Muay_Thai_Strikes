import streamlit as st
import numpy as np
import imutils
import time
import os
import cv2 as cv
import tempfile
import configure

st.title('Muay Thai Video Classification App')
st.text('Built with Streamlit,Yolov3 and OpenCV')


confidence_threshold, nms_threshold = 0.5, 0.3
net = cv.dnn.readNetFromDarknet(configure.CONFIG_PATH, configure.MODEL_PATH)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]


f = st.file_uploader("Upload file", type=['mp4'])

if st.button("Identify"):
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(f.read())

    vf = cv.VideoCapture(tfile.name)

    stframe = st.empty()

    (W,H) = (None, None)

    while vf.isOpened():
        grabbed, frame = vf.read()

        # if frame is read correctly ret is True
        if not grabbed:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        if W is None or H is None:
            H,W = frame.shape[:2]

        blob = cv.dnn.blobFromImage(frame, 1/255.0, (416,416), swapRB = True, crop = False)
        start = time.time()
        net.setInput(blob)
        layerOutputs = net.forward(ln)
        end = time.time()

        boxes = []
        confidences = []
        classIDs = []

        # loop over each output from layeroutputs
        for output in layerOutputs:
            # loop over each detecton in output
            for detection in output:
                # extract score, ids and confidence of current object detection
                score = detection[5:]
                classID = np.argmax(score)
                confidence = score[classID]

                # filter out weak detections with confidence threshold
                if confidence > confidence_threshold:
                    # scale bounding box coordinates back relative to image size
                    # YOLO spits out center (x,y) of bounding boxes followed by 
                    # boxes width and heigth
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype('int')

                    # grab top left coordinate of the box
                    x = int(centerX - (width/2))
                    y = int(centerY - (height/2))

                    boxes.append([x,y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)


    # Apply Non-Max Suppression, draw boxes and write output video 

        idxs = cv.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)
        # ensure detection exists
        if len(idxs) > 0:
            for i in idxs.flatten():
                # getting box coordinates
                (x,y) = (boxes[i][0], boxes[i][1])
                (w,h) = (boxes[i][2], boxes[i][3])

                # color and draw boxes
                color = [int(c) for c in configure.COLORS[classIDs[i]]]
                cv.rectangle(frame, (x,y), (x+w, y+h), color, 2)
                text = f"{configure.LABELS[classIDs[i]]}: {confidences[i]}"
                cv.putText(frame, text, (x, y-5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)




            color = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            stframe.image(color)





