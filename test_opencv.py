import numpy as np
import cv2

with open('coco.names', 'r') as f:
  classes = [line.strip() for line in f.readlines()]

image = cv2.imread('apple-single-red.jpg')
blob = cv2.dnn.blobFromImage(image, 1 / 255, (416, 416), (0, 0, 0), True, crop=False)
# Load the network
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
# set as input to the net
net.setInput(blob)
# get network output layers
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
# inference
# the network outputs multiple lists of anchor boxes,
# one for each detected class
outs = net.forward(output_layers)
# extract bounding boxes
class_ids = list()
confidences = list()
boxes = list()
# iterate over all classes
for out in outs:
    # iterate over the anchor boxes for each class
    for detection in out:
        # bounding box
        center_x = int(detection[0] * image.shape[1])
        center_y = int(detection[1] * image.shape[0])
        w = int(detection[2] * image.shape[1])
        h = int(detection[3] * image.shape[0])
        x = center_x - w // 2
        y = center_y - h // 2
        boxes.append([x, y, w, h])
        # class
        class_id = np.argmax(detection[5:])
        class_ids.append(class_id)
        # confidence
        confidence = detection[4]
        confidences.append(float(confidence))
# non-max suppression
ids = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.3, nms_threshold=0.5)
# draw the bounding boxes on the image
colors = np.random.uniform(0, 255, size=(len(classes), 3))
for i in ids:
    i = i[0]
    x, y, w, h = boxes[i]
    class_id = class_ids[i]
    color = colors[class_id]
    cv2.rectangle(image, (round(x), round(y)), (round(x + w), round(y + h)), color, 2)
    label = "%s: %.2f" % (classes[class_id], confidences[i])
    cv2.putText(image, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1)
cv2.imshow("Object detection", image)
cv2.waitKey()
# download weight and cfg from offical yolo git or website
# load your img to net and see , verry 
