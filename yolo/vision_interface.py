from abc import abstractmethod
import cv2
import tensorflow as tf
from scipy.spatial.distance import cityblock
import numpy as np

class TF_classifier:
    @abstractmethod
    def __init__(self, path):
        interpreter = tf.lite.Interpreter(model_path=path)
        interpreter.allocate_tensors()
        self.interpreter =  interpreter

    @abstractmethod
    def prep_image(self, sd_train):
        pass

    @abstractmethod
    def predict(self, img):
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()

        self.interpreter.set_tensor(input_details[0]['index'], img)
        self.interpreter.invoke()
        
        return self.interpreter.get_tensor(output_details[0]['index'])

    @abstractmethod
    def adapt(self, result):
        pass

    @abstractmethod
    def run(self, img):
        #prepare the image
        prep_img = self.prep_image(img)
        #give the prepared image to the model and save its output
        result = self.predict(prep_img)
        #adapter function to make a uniqe value from the result
        return self.adapt(result)

class Seeger_DM_Yolo(TF_classifier):
    def __init__(self, path):
        super().__init__(path)       

    def prep_image(self, img):
        self.img = img
        img = cv2.resize(img, dsize=(640, 640), interpolation=cv2.INTER_CUBIC)
        img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
        img = img.astype(np.float32)
        img /= 255
        return img

    def predict(self, img):
        return super().predict(img)


    '''
    MAKE A BOOLEAN VALUE FROM THE MEAN OF THE DISTANCES BETWEEN KOORDINATES
    '''
    def adapt(self, result):
        xyxy, classes, scores = self.YOLOdetect(result) #boxes(x,y,x,y), classes(int), scores(float) [25200]

        koords = []
        for i in range(len(scores)):
                #treshold values 
                if ((scores[i] > 0.5) and (scores[i] <= 1.0)):
                    H = self.img.shape[0]
                    W = self.img.shape[1]
                    xmin = int(max(1,(xyxy[0][i] * W)))
                    ymin = int(max(1,(xyxy[1][i] * H)))
                    xmax = int(min(W,(xyxy[2][i] * W)))
                    ymax = int(min(H,(xyxy[3][i] * H)))

                    koords.append(((xmin,ymin), (xmax, ymax)))

        
        distances = self.dist_calc(koords)
        return True if np.mean(distances) < 200 else False

    def run(self, img):
        return super().run(img)

    '''
    CALCULATE THE DISTANCE BY THE KOORDINATES
    '''
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

        return distances

    '''
    OUTPUT CONVERTER FUNCTIONS
    '''
    def classFilter(self, classdata):
        classes = ["seeger"]  # create a list
        for i in range(classdata.shape[0]):         # loop through all predictions
            classes.append(classdata[i].argmax())   # get the best classification location
        return classes  # return classes (int)

    def YOLOdetect(self, output_data):  # input = interpreter, output is boxes(xyxy), classes, scores
        output_data = output_data[0]                # x(1, 25200, 7) to x(25200, 7)
        boxes = np.squeeze(output_data[..., :4])    # boxes  [25200, 4]
        scores = np.squeeze( output_data[..., 4:5]) # confidences  [25200, 1]
        classes = self.classFilter(output_data[..., 5:]) # get classes
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        x, y, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3] #xywh
        xyxy = [x - w / 2, y - h / 2, x + w / 2, y + h / 2]  # xywh to xyxy   [4, 25200]

        return xyxy, classes, scores
