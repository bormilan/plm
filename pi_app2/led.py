import RPi.GPIO as GPIO
import time
from bse.logger import get_logger


class LED:
    def __init__(self, state = False):
        self.state = state
        self.logger = get_logger('LED')
        try:
            GPIO.setwarnings(False)
            GPIO.setmode(GPIO.BOARD)
            GPIO.setup(11, GPIO.OUT)
            GPIO.setup(13, GPIO.OUT)
        except:
            self.logger.debug("GPIO already set up.")
        
    def switch(self, value):
        self.state = value
        if value:
            GPIO.output(11, True)
            GPIO.output(13, True)
            self.logger.debug("LED on")
        else:
            GPIO.output(11, False)
            GPIO.output(13, False)
            self.logger.debug("LED off")
    
    def cleanup(self):
        GPIO.cleanup()