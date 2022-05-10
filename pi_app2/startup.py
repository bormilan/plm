import RPi.GPIO as GPIO # Import Raspberry Pi GPIO library
import dist_measure as dm

model = ""

def prediction_callback(channel):
    # distance = dm.distance_measurement(model, path_image)
    print('1')
    
def save_callback(channel):
    print("2")

def setup():
    GPIO.setwarnings(False) # Ignore warning for now
    GPIO.setmode(GPIO.BOARD) # Use physical pin numbering
    GPIO.setup(10, GPIO.IN, pull_up_down=GPIO.PUD_DOWN) # Set pin 10 to be an input pin and set initial value to be pulled low (off)
    GPIO.setup(12, GPIO.IN, pull_up_down=GPIO.PUD_DOWN) # Set pin 10 to be an input pin and set initial value to be pulled low (off)

    GPIO.add_event_detect(10,GPIO.RISING,callback=prediction_callback) # Setup event on pin 10 rising edge
    GPIO.add_event_detect(12,GPIO.RISING,callback=save_callback) # Setup event on pin 10 rising edge

    # global model 
    # model = dm.startup("model_knn.sav")

def cleanup():
    GPIO.cleanup() # Clean up