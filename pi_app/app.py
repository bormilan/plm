import cv2
from pandas import set_option
from flask import Response, request

from startup import app, model
from functions import get_frame, decode_command

import socket

# from led import LED

@app.route('/')
def index():
    return open("index.html", "r").read()

@app.route('/video_feed')
def video_feed():
    return Response(get_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/controll', methods = ['POST', 'GET'])
def controll():
    if request.method == 'POST':        
        result = decode_command(request.form['value'])
        print(result)
        return result, 200
    elif request.method == 'GET':        
        return "Ez nem erre valo", 200

if __name__ == '__main__':
    # Find local ip
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    ip = s.getsockname()[0]
    s.close()
    # use ip variable when connected to lan
    app.run(host=ip, port=80, threaded=True)