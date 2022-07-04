import cv2
import tensorflow as tf
import numpy as np

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

    return xyxy, classes, scores  # output is boxes(x,y,x,y), classes(int), scores(float) [predictions length]

def cam():
    interpreter = tf.lite.Interpreter(model_path="seeger_yolov2.tflite")
    interpreter.allocate_tensors()

    video = cv2.VideoCapture(0)

    font = cv2.FONT_HERSHEY_SIMPLEX
    
    
    # img = cv2.imread("test.jpg") 

    while True:
        success, frame = video.read()

        if success:

            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            img = cv2.resize(frame, dsize=(640, 640), interpolation=cv2.INTER_CUBIC)
            img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
            img = img.astype(np.float32)
            img /= 255

            interpreter.set_tensor(input_details[0]['index'], img)
            interpreter.invoke()
            result = interpreter.get_tensor(output_details[0]['index'])
            xyxy, classes, scores = YOLOdetect(result) #boxes(x,y,x,y), classes(int), scores(float) [25200]

            for i in range(len(scores)):
                #treshold values 
                if ((scores[i] > 0.5) and (scores[i] <= 1.0)):
                    H = frame.shape[0]
                    W = frame.shape[1]
                    xmin = int(max(1,(xyxy[0][i] * W)))
                    ymin = int(max(1,(xyxy[1][i] * H)))
                    xmax = int(min(H,(xyxy[2][i] * W)))
                    ymax = int(min(W,(xyxy[3][i] * H)))

                    cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
                    cv2.putText(frame, str(scores[i]), (xmin,ymin), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
            
            cv2.imshow('test', frame)
            if cv2.waitKey(1) == ord('q'):
                cv2.destroyAllWindows()
                break

def image(path):

    image = cv2.imread(path)

    interpreter = tf.lite.Interpreter(model_path="seeger_yolov2.tflite")
    interpreter.allocate_tensors()

    font = cv2.FONT_HERSHEY_SIMPLEX

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    img = cv2.resize(image, dsize=(640, 640), interpolation=cv2.INTER_CUBIC)
    img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
    img = img.astype(np.float32)
    img /= 255

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    result = interpreter.get_tensor(output_details[0]['index'])
    xyxy, classes, scores = YOLOdetect(result) #boxes(x,y,x,y), classes(int), scores(float) [25200]

    for i in range(len(scores)):
        #treshold values 
        if ((scores[i] > 0.1) and (scores[i] <= 1.0)):
            H = image.shape[0]
            W = image.shape[1]
            xmin = int(max(1,(xyxy[0][i] * W)))
            ymin = int(max(1,(xyxy[1][i] * H)))
            xmax = int(min(W,(xyxy[2][i] * W)))
            ymax = int(min(H,(xyxy[3][i] * H)))

            cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
            cv2.putText(image, str(scores[i]), (xmin,ymin), font, 1, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow("result", image)
    cv2.waitKey(0)

# import time
def main():
    
    while True:
        type = input("pease enter the mode (cam or image):")
        if type == "cam":
            cam()
            break
        elif type == "image":
            path = input("enter the path of the image:")
            # start_time = time.time()
            image(path)
            # print("--- %s seconds ---" % (time.time() - start_time))  
            
            break
        else:
            print("only cam or image are the valid modes")

main()