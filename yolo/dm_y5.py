import cv2
import tensorflow as tf
import numpy as np
from scipy.spatial.distance import cityblock

def classFilter(classdata):
    classes = ["seeger"]  # create a list
    for i in range(classdata.shape[0]):         # loop through all predictions
        classes.append(classdata[i].argmax())   # get the best classification location
    return classes  # return classes (int)

def YOLOdetect(output_data):  # input = interpreter, output is boxes(xyxy), classes, scores
    output_data = output_data[0]                # x(1, 25200, 7) to x(25200, 7)
    boxes = np.squeeze(output_data[..., :4])    # boxes  [25200, 4]
    scores = np.squeeze( output_data[..., 4:5]) # confidences  [25200, 1]
    classes = classFilter(output_data[..., 5:]) # get classes
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    x, y, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3] #xywh
    xyxy = [x - w / 2, y - h / 2, x + w / 2, y + h / 2]  # xywh to xyxy   [4, 25200]

    return xyxy, classes, scores

class Yolo():
    def __init__(self, path):
        interpreter = tf.lite.Interpreter(model_path=path)
        interpreter.allocate_tensors()
        self.interpreter =  interpreter

    def prep_img(self, img):
        self.img = img
        img = cv2.resize(img, dsize=(640, 640), interpolation=cv2.INTER_CUBIC)
        img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
        img = img.astype(np.float32)
        img /= 255
        return img

    def predict(self, img):
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()

        self.interpreter.set_tensor(input_details[0]['index'], img)
        self.interpreter.invoke()
        result = self.interpreter.get_tensor(output_details[0]['index'])
        xyxy, classes, scores = YOLOdetect(result) #boxes(x,y,x,y), classes(int), scores(float) [25200]

        koords = []
        for i in range(len(scores)):
                #treshold values 
                if ((scores[i] > 0.5) and (scores[i] <= 1.0)):
                    H = self.img.shape[0]
                    W = self.img.shape[1]
                    xmin = int(max(1,(xyxy[0][i] * W)))
                    ymin = int(max(1,(xyxy[1][i] * H)))
                    #ennél a kettőnél fordítva van a W és a H a stackowerflow-ban
                    xmax = int(min(W,(xyxy[2][i] * W)))
                    ymax = int(min(H,(xyxy[3][i] * H)))

                    # cv2.rectangle(self.img, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
                    koords.append(((xmin,ymin), (xmax, ymax)))
        # cv2.imshow("result", self.img)
        # cv2.waitKey(0)

        return koords

    def dist_calc(self, koords):
        a_koords = [koords[0]]
        b_koords = []

        for k in koords[1:len(koords)]:
            l = a_koords[0]
            o1 = int((k[0][0] + k[1][0]) / 2), int((k[0][1] + k[1][1]) / 2)
            o2 = int((l[0][0] + l[1][0]) / 2), int((l[0][1] + l[1][1]) / 2)

            # print(np.linalg.norm(np.array(o2) - np.array(o1)))
            if np.linalg.norm(np.array(o2) - np.array(o1)) < 50:
                a_koords.append(k)
            else:
                b_koords.append(k)

        distances = []
        for a, b in zip(a_koords, b_koords):
            o1 = int((a[0][0] + a[1][0]) / 2), int((a[0][1] + a[1][1]) / 2)
            o2 = int((b[0][0] + b[1][0]) / 2), int((b[0][1] + b[1][1]) / 2)

            distances.append(cityblock(o1, o2))

        return np.mean(distances)

            # print(len(a_koords))
            # print(len(b_koords))

        # image = self.img
        # for k in b_koords:
        #     o = int((k[0][0] + k[1][0]) / 2), int((k[0][1] + k[1][1]) / 2)
        #     cv2.circle(image, o, radius=0, color=(0, 0, 255), thickness=-1)

        # cv2.imshow("result", image)
        # cv2.waitKey(0)
        

def main():
    model = Yolo("seeger_yolov2.tflite")
    img = cv2.imread("test.jpg")

    test_images = make_data_from_folder('/Users/bormilan/Documents/plm_kepek_detect/test_raw')
    # test_images = make_data_from_folder('/Users/bormilan/Documents/plm_kepek_detect/new_knorr/photos_1')
    # test_images = make_data_from_folder('/Users/bormilan/Documents/plm_kepek_detect/raw_incorrect')

    # cv2.imshow("result", model.predict(model.prep_img(img)))
    for img in test_images:
        koords = model.predict(model.prep_img(img))
        dist = model.dist_calc(koords)
        
        # for k in koords:
        #     o = int((k[0][0] + k[1][0]) / 2), int((k[0][1] + k[1][1]) / 2)
        #     cv2.circle(img, o, radius=0, color=(0, 0, 255), thickness=-1)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(dist), (50,50), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow("result", img)
        cv2.waitKey(0)

        print(dist)
    cv2.waitKey(0)

import os
def make_data_from_folder(path):
  data = []
  for img in os.listdir(path):
    if img != ".DS_Store":
      pic = cv2.imread(os.path.join(path,img))
      pic = cv2.cvtColor(pic,cv2.COLOR_BGR2RGB)
      # pic = cv2.resize(pic,(80,80))
      data.append(pic)
  return data

main()