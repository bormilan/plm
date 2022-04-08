import cv2
from abc import abstractmethod
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
import os
from datetime import datetime
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array
import numpy as np
import joblib
from skimage import feature


class Camera:
    def __init__(self):      
        width = 1080
        height = 720
        self.video = cv2.VideoCapture(0)       
        self.video.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.video.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        self.video.set(cv2.CAP_PROP_AUTOFOCUS, 1) # auto

        # self.led = LED()

class Classifier:
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def load(self, sd_train):
        pass

    @abstractmethod
    def prep_image(self, sd_train):
        pass

    @abstractmethod
    def predict(self, sd_test):
        pass

    @abstractmethod
    def save(self):
        pass

class KNN(Classifier):
    def __init__(self, path):
        self.model = KNeighborsClassifier(n_neighbors=1)

    def load(self, path):
        self.model = joblib.load(path)

    def prep_image(img):
        img = automatic_brightness_and_contrast(img, 10)
        img = img.crop((300, 300, 700, 700))
        img = img_to_array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return img

    def predict(self, img):
        result_strings = ["helyes felhelyezés", "jobb fog feljebb van", "bal fog feljebb van", "az egész feljebb van"]

        img = self.prep_img(img)
        img_hog = feature.hog(img)

        result = self.model.predict(img_hog.reshape(1, -1))
        return result_strings[int(result)]

class SVC(Classifier):
    def __init__(self):
        self.model = SVC(kernel="linear")

    def load(self, path):
        self.model = joblib.load(path)

    def prep_image(img):
        img = automatic_brightness_and_contrast(img, 10)
        img = img.crop((300, 300, 700, 700))
        img = img_to_array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) 

    def predict(self, img):
        result_strings = ["helyes felhelyezés", "jobb fog feljebb van", "bal fog feljebb van", "az egész feljebb van"]

        img = self.prep_img(img)
        img_hog = feature.hog(img)

        result = self.model.predict(img_hog.reshape(1, -1))
        return result_strings[int(result)]

class CNN(Classifier):
    def __init__(self):
        interpreter = tf.lite.Interpreter(model_path="lite_cnn_model.tflite")
        interpreter.allocate_tensors()
        self.model =  interpreter

    def prep_img(img):
        img = array_to_img(img)
        img = img.crop((300, 300, 700, 700))
        img = (img_to_array(img)).astype('float32')
        img = cv2.resize(img, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
        img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))

    def predict(self, img):
        result_strings = ["helyes felhelyezés", "jobb fog feljebb van", "bal fog feljebb van", "az egész feljebb van"]

        img = self.prep_img(img)
        cv2.imwrite("photos/"+str(datetime.datetime.now()).replace(".","-")+'.jpg', img[0])
        # Get input and output tensors.
        input_details = self.model.get_input_details()
        output_details = self.model.get_output_details()
        
        self.model.set_tensor(input_details[0]['index'], img)
        self.model.invoke()
        return result_strings[np.argmax(self.model.get_tensor(output_details[0]['index'])[0])]

def automatic_brightness_and_contrast(img, clip_hist_percent):
    img = img_to_array(img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Calculate grayscale histogram
    hist = cv2.calcHist([gray],[0],None,[256],[0,256])
    hist_size = len(hist)
    
    # Calculate cumulative distribution from the histogram
    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index -1] + float(hist[index]))
    
    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum/100.0)
    clip_hist_percent /= 2.0
    
    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1
    
    # Locate right cut
    maximum_gray = hist_size -1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1
    
    # Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha
    

    auto_result = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return array_to_img(auto_result)