import cv2
from classes import Camera, KNN, CNN, SVC
from startup import model

def get_frame():
    cam = Camera()

    while True:
        success, image = cam.video.read() 

        ret, jpeg = cv2.imencode('.jpg', image)
        frame = jpeg.tobytes()                
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def get_one_frame():
    cam = Camera()

    success, image = cam.video.read() 
    if success:
        return image

def prediction():
    print("ide is")
    return model.predict(get_one_frame())

def set_mode(cmd):
    if cmd == "knn":
        model = KNN()
        model.load("/Users/bormilan/Documents/kód/plm/finalized_model_knn3.sav")
    if cmd == "svc":
        model = SVC()
        model.load("")
    if cmd == "cnn":
        model = CNN()
        model.load("")
    return

def decode_command(cmd):
    if cmd == "q":
        return "-"
    elif cmd == "p":
        return prediction()
    else:
        try:
            set_mode(cmd)
            return "sikeres modell váltás"
        except:
            return "sikertelen modell váltás"