import os
import glob
import cv2
from skimage import feature
import pickle
import numpy as np
import dlib
from sklearn.preprocessing import LabelEncoder


i = 0
BINS = 9
BLOCK_SIZE = np.array([2,2])  # cell
CELL_SIZE = np.array([10,10]) # pixel
NORM = "L2"


# tính tọa độ face
def hog_face_to_points(rect):
	x1 = rect.left()
	y1 = rect.top()
	x2 = rect.right()
	y2 = rect.bottom()

	return x1, y1, x2, y2

# detect face
def _detect_face(img):
    hog_face_detector = dlib.get_frontal_face_detector()
    faces = hog_face_detector(img, upsample_num_times=0)
    ls = []
    for face in faces :
        x1, y1, x2, y2 = hog_face_to_points(face)
        ls.append(x1)
        ls.append(y1)
        ls.append(x2)
        ls.append(x2)

    return ls

# trích xuất đặc trưng hog
def _preprocessing(fileType):
    data = []
    labels = []
    for path in glob.glob(fileType):
        _, brand, fn = path.split('\\')
        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ls = _detect_face(gray)
        if(len(ls)):
            [x1, y1, x2, y2] = ls
            logo = gray[y1:y2, x1:x2]
            logo = cv2.resize(logo, (200, 200))
            # Khởi tạo HOG descriptor
            H = feature.hog(logo, orientations=BINS, pixels_per_cell=CELL_SIZE,
                            cells_per_block=BLOCK_SIZE, transform_sqrt=True, block_norm=NORM)
            # update the data and labels
            data.append(H)
            labels.append(brand)
    return data, labels

data_train, label_train = _preprocessing('images/train/**/*.jpg')
data_test, label_test  = _preprocessing('images/test/**/*.jpg')

def _save(path, obj):
    with open(path, 'wb') as fn:
        pickle.dump(obj, fn)

# transform dữ liệu
def _transform_data(data, labels):
    # Tạo input array X
    X = np.array(data)
    # Tạo output array y
    le = LabelEncoder()
    le.fit(labels)
    y = le.transform(labels)
    y_ind = np.unique(y)
    y_dict = dict(zip(y_ind, le.classes_))
    return X, y, y_dict, le

def _accuracy(y_pred, y):
    temp = np.array(y_pred) - np.array(y)
    N_false = np.count_nonzero(temp)
    N = len(y)
    return 100 *(N-N_false)/N

X_train, y_train, y_train_dict, le = _transform_data(data_train, label_train)
X_test, y_test, y_test_dict, le_test = _transform_data(data_test, label_test)