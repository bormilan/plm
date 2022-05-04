import cv2
import command_center as cc
import dist_measure as dm

def main():
	cap = cv2.VideoCapture(0)
	# cv2.namedWindow("test", cv2.WND_PROP_FULLSCREEN)
	# cv2.setWindowProperty("test", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

	dm_model = dm.Distance_measure("model_knn.sav", 200)
	command_center = cc.Command_center(dm_model)
	command_center.setup()

	# img1 = cv2.imread("2022-04-11 11_35_43-110600.jpg")
	# img2 = cv2.imread("2022-04-12 14_45_36-759277.jpg")

	print("setup done")

	try:
		while True:
			success, frame = cap.read()

			if success:
				command_center.set_img(frame)

			# cv2.imshow('test', frame)

			# if i % 2 == 0:
			# 	command_center.set_img(img1)
			# else:
			# 	command_center.set_img(img2)
	except KeyboardInterrupt:	
		command_center.cleanup()
		cap.release()
	# cv2.destroyAllWindows()

main()