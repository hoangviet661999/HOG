import cv2
import dlib
import imutils
import matplotlib.pyplot as plt

def hog_face_to_points(rect):
	x1 = rect.left()
	y1 = rect.top()
	x2 = rect.right()
	y2 = rect.bottom()

	return x1, y1, x2, y2

image_path='images/train/abert enstein/enstein1.jpg'

image = cv2.imread(image_path)

rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

hog_face_detector = dlib.get_frontal_face_detector()

faces = hog_face_detector(rgb_image, upsample_num_times=0)

green_color = (0, 255, 0)

for face in faces:
	x1, y1, x2, y2 = hog_face_to_points(face)
	cv2.rectangle(image, pt1=(x1,y1), pt2=(x2,y2), color=green_color, thickness=1)
	cv2.imshow('image', image[y1:y2, x1:x2])
	ls = [x1,y1,x2,y2]
	print([x2-x1, y2-y1])
	cv2.imshow('img', image)
	cv2.waitKey(0)




