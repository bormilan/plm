import cv2
import command_center as cc
import dist_measure as dm
from led import LED
import time

def main():
	
	video = cv2.VideoCapture(0)
	width = 1080
	height = 720
	video.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc('M','J','P','G'))
	video.set(cv2.CAP_PROP_BUFFERSIZE, 1)
	video.set(cv2.CAP_PROP_FRAME_WIDTH, width)
	video.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
	video.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
	# cv2.namedWindow("test", cv2.WND_PROP_FULLSCREEN)
	# cv2.setWindowProperty("test", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

	dm_model = dm.Distance_measure("model_knn.sav", 215)
	command_center = cc.Command_center(dm_model)
	command_center.setup()

	# img1 = cv2.imread("2022-04-11 11_35_43-110600.jpg")
	# img2 = cv2.imread("2022-04-12 14_45_36-759277.jpg")

	led = LED()
	led.switch(True)

	print("setup done")
	font = cv2.FONT_HERSHEY_PLAIN  

	while True:
		success, frame = video.read()
		
		# if success:
		command_center.set_img(frame)

		#pred text
		if command_center.last_pred != None and (time.time() - command_center.last_pred) > 5:
			command_center.text_pred = None
			#if correct, the color is green, if incorrect, its red
		color = (0, 0, 255)
		if command_center.text_pred == "Helyes felhelyezes":
			color = (0, 255, 0)
		cv2.putText(frame, command_center.text_pred, (100, 100), font, 2, color, 3, cv2.LINE_AA)				

		#photo text
		if command_center.last_photo != None and (time.time() - command_center.last_photo) > 1:
			command_center.text_photo = None
	 
		cv2.putText(frame, command_center.text_photo, (100, 300), font, 2, (0, 0, 255), 3, cv2.LINE_AA)

		cv2.imshow('test', frame)

		if cv2.waitKey(1) == ord('q'):
			command_center.cleanup()
			video.release()
	# cv2.destroyAllWindows()
	
main()