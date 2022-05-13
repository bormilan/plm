import RPi.GPIO as GPIO # Import Raspberry Pi GPIO library
import cv2
import datetime
import time

class Command_center:

    measure_obj = None
    img = None
    text_pred = None
    last_pred = None
    text_photo = None
    last_photo = None

    def __init__(self, dm):
        self.measure_obj = dm

    def setup(self):
        GPIO.setwarnings(False) # Ignore warning for now
        GPIO.setmode(GPIO.BOARD) # Use physical pin numbering
        GPIO.setup(10, GPIO.IN, pull_up_down=GPIO.PUD_DOWN) # Set pin 10 to be an input pin and set initial value to be pulled low (off)
        GPIO.setup(12, GPIO.IN, pull_up_down=GPIO.PUD_DOWN) # Set pin 10 to be an input pin and set initial value to be pulled low (off)

        GPIO.add_event_detect(10,GPIO.RISING,callback=self.prediction_callback, bouncetime = 500) # Setup event on pin 10 rising edge
        GPIO.add_event_detect(12,GPIO.RISING,callback=self.save_callback, bouncetime = 500)

    def set_img(self, img):
        self.img = img

    def cleanup(self):
        GPIO.cleanup() # Clean up

    def prediction_callback(self, channel): 
        result = self.measure_obj.distance_measurement(self.img)
        self.text_pred = result
        self.last_pred = time.time()
        
    def save_callback(self, channel):
        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        img = img[200:440, 300:700]
        cv2.imwrite("photos/"+str(datetime.datetime.now()).replace(".","-")+'.jpg', img)
        self.text_photo = "sikeres mentes"
        self.last_photo = time.time()
