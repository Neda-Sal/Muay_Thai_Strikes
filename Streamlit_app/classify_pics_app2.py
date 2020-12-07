import streamlit as st
import cv2 as cv
import numpy as np
import time
from PIL import Image


################################################################################
WHITE = (255, 255, 255)
img = None
img0 = None
outputs = None

################################################################################
# Load names of classes and get random colors

classes = ["kick", "punch", "knee", "elbow", "clinch", "stance"]
np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(classes), 4), dtype='uint8')

# Give the configuration and weight files for the model and load the network.
net = cv.dnn.readNetFromDarknet('yolov3_testing.cfg', 'yolov3_training_12000.weights')
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)


# determine the output layer
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

################################################################################
### Functions


def load_image(img_array):
    # st.write('Loading image...')
    global img, img0, outputs, output_layers
    
    #img0 = cv.imread(path) # change this to work with output of streamlit file uploader
    img0 = img_array
    img = img0.copy()
    blob = cv.dnn.blobFromImage(img, 1/255.0, (416, 416),swapRB=True, crop=False)
    net.setInput(blob)
    t0 = time.time()
    outputs = net.forward(output_layers)
    t = time.time() - t0
    
    # combine the 3 output groups into 1 (10647, 85)
    # large objects (507, 85)
    # medium objects (2028, 85)
    # small objects (8112, 85)
    outputs = np.vstack(outputs)
    post_process(img, outputs, 0.5)
    st.image(img.astype(np.uint8))
    
def post_process(img, outputs, conf):
    # st.write('Post-processing...')
    H, W = img.shape[:2]
    boxes = []
    confidences = []
    classIDs = []
    for output in outputs:
        scores = output[5:]
        classID = np.argmax(scores)
        confidence = scores[classID]
        if confidence > conf:
            x, y, w, h = output[:4] * np.array([W, H, W, H])
            p0 = int(x - w//2), int(y - h//2)
            p1 = int(x + w//2), int(y + h//2)
            boxes.append([*p0, int(w), int(h)])
            confidences.append(float(confidence))
            classIDs.append(classID)
            # cv.rectangle(img, p0, p1, WHITE, 1)
    indices = cv.dnn.NMSBoxes(boxes, confidences, conf, conf-0.1)
    if len(indices) > 0:
        for i in indices.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            color = [int(c) for c in colors[classIDs[i]]]
            cv.rectangle(img, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(classes[classIDs[i]], confidences[i])
            cv.putText(img, text, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            st.write(classes[classIDs[i]], confidences[i])
def trackbar(x):
    global img
    conf = x/100
    img = img0.copy()
    post_process(img, outputs, conf)
    cv.displayOverlay('window', f'confidence level={conf}')
    cv.imshow('window', img)
    
################################################################################

### Run model

st.title('Muay Thai Classification App')
st.text('Built with Streamlit,Yolov3 and OpenCV')

menu = ['Detection','About']
choice = st.sidebar.selectbox('Menu',menu)

if choice == 'Detection':
    st.subheader('Strike Detection')

    img_file_buffer = st.file_uploader("Upload Image",type=['jpg', 'png', 'jpeg'])

    if img_file_buffer is not None:
        our_image = Image.open(img_file_buffer)
        st.text('Original Image')
        # st.write(type(our_image))
        st.image(our_image)
            
    if st.button("Identify"):
            image = Image.open(img_file_buffer)
            img_array = np.array(image)
            load_image(img_array)
            st.write('did it!')
            
elif choice == 'About':
    st.subheader('About')
    st.info('Muay Thai is a standing martial art, often called "The art of 8 limbs", that utilizes punching, elbowing, kneeing, and kicking an opponent.')

