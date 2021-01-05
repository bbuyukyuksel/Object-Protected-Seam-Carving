import numpy as np
import cv2
import pathlib

class YoloV4:
    __config_dir = "yolov4config"
    __weights = pathlib.Path(__config_dir) / "yolov4.weights"
    __cfg = pathlib.Path(__config_dir) / "yolov4.cfg"
    __coco_names = pathlib.Path(__config_dir) / "coco.names"

    def __init__(self, threshold=0.85):
        assert self.__weights.exists()
        assert self.__cfg.exists()
        assert self.__coco_names

        with open(self.__coco_names, 'r') as f:
            self.__classes = [line.strip() for line in f.readlines()]

        self.threshold = threshold

        self.net = cv2.dnn.readNet(str(self.__weights), str(self.__cfg))
        layer_names = self.net.getLayerNames()
        self.output_layers = [layer_names[i[0]-1] for i in self.net.getUnconnectedOutLayers()]

        

    def predict(self, img):
        H, W, channels = img.shape

        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0,0,0), True, crop=False)
        self.net.setInput(blob)
        layerOutputs  = self.net.forward(self.output_layers)

        boxes = []
        confidences = []
        classIDs = []
        for output in layerOutputs:
        # loop over each of the detections
            for detection in output:
                # extract the class ID and confidence (i.e., probability) of
                # the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                # filter out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence > self.threshold:
                    # scale the bounding box coordinates back relative to the
                    # size of the image, keeping in mind that YOLO actually
                    # returns the center (x, y)-coordinates of the bounding
                    # box followed by the boxes' width and height
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    # use the center (x, y)-coordinates to derive the top and
                    # and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    # update our list of bounding box coordinates, confidences,
                    # and class IDs
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence, self.threshold)
        return idxs, boxes

    def getBBOXs(self, img, visualize=False):
        BBOXs = {
            "bboxs" : []
        }
        idxs, bboxs = self.predict(img)
        for id in idxs:
            x,y,w,h = bboxs[np.squeeze(id)]
            BBOXs["bboxs"].append([x,y,w,h])
            if visualize: cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 1)    
        
        if visualize:
            cv2.imshow("Show:BBOXs", img)
            cv2.waitKey(2000)
            cv2.destroyWindow("Show:BBOXs")

        return BBOXs

if __name__ == '__main__':
    img = cv2.imread('Images/image.jpg')
    yolov4 = YoloV4()
    bboxs = yolov4.getBBOXs(img, visualize=True)
    print(bboxs["bboxs"])
    
