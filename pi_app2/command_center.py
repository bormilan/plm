import RPi.GPIO as GPIO # Import Raspberry Pi GPIO library
import dist_measure as dm
import cv2
import datetime
import numpy as np

class Command_center:

    measure_obj = None
    img = None

    def __init__(self, dm):
        self.measure_obj = dm

    def setup(self):
        GPIO.setwarnings(False) # Ignore warning for now
        GPIO.setmode(GPIO.BOARD) # Use physical pin numbering
        GPIO.setup(10, GPIO.IN, pull_up_down=GPIO.PUD_DOWN) # Set pin 10 to be an input pin and set initial value to be pulled low (off)
        GPIO.setup(12, GPIO.IN, pull_up_down=GPIO.PUD_DOWN) # Set pin 10 to be an input pin and set initial value to be pulled low (off)

        GPIO.add_event_detect(10,GPIO.RISING,callback=self.prediction_callback) # Setup event on pin 10 rising edge
        GPIO.add_event_detect(12,GPIO.RISING,callback=self.save_callback)

    def set_img(self, img):
        self.img = img

    def cleanup(self):
        GPIO.cleanup() # Clean up

    def prediction_callback(self, channel):
        distance = self.measure_obj.distance_measurement(self.img)
        print(distance)
        
    def save_callback(self, channel):
        cv2.imwrite("photos/"+str(datetime.datetime.now()).replace(".","-")+'.jpg', self.img)
        print("ye")