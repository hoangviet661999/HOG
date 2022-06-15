import cv2
import dlib

def hog_face_to_points(rect):
	x1 = rect.left()
	y1 = rect.top()
	x2 = rect.right()
	y2 = rect.bottom()

	return x1, y1, x2, y2

image_path='images/kimoanh.bmp'

image = cv2.imread(image_path)

rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# load mô hình nhận diện khuôn mặt
hog_face_detector = dlib.get_frontal_face_detector()
# nhận diện khuôn mặt trong ảnh
faces = hog_face_detector(rgb_image, upsample_num_times=0)
# vẽ đường bao cho từng khuôn mặt
green_color = (0, 255, 0)
for face in faces:
    x1, y1, x2, y2 = hog_face_to_points(face)
    cv2.rectangle(image, pt1=(x1, y1), pt2=(x2, y2), color=green_color, thickness=1)
    cv2.imshow('img', image)
    cv2.waitKey(0)