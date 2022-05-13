import RPi.GPIO as GPIO


class LED:
    def __init__(self, state = False):
        self.state = state
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(11, GPIO.OUT)
        GPIO.setup(13, GPIO.OUT)
        
    def switch(self, value):
        self.state = value
        if value:
            GPIO.output(11, True)
            GPIO.output(13, True)
        else:
            GPIO.output(11, False)
            GPIO.output(13, False)
    
    def cleanup(self):
        GPIO.cleanup()