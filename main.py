import os
import glob
import cv2
from skimage import feature
import pickle
from sklearn.preprocessing import LabelEncoder
import numpy as np
import dlib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

i = 0
# tính tọa độ face
def hog_face_to_points(rect):
	x1 = rect.left()
	y1 = rect.top()
	x2 = rect.right()
	y2 = rect.bottom()

	return x1, y1, x2, y2

# detect face
def detect_face(img):
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
        ls = detect_face(gray)
        if(len(ls)):
            [x1, y1, x2, y2] = ls
            logo = gray[y1:y2, x1:x2]
            logo = cv2.resize(logo, (200, 200))
            # Khởi tạo HOG descriptor
            H = feature.hog(logo, orientations=9, pixels_per_cell=(10, 10),
                            cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1")

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

X_train, y_train, y_train_dict, le = _transform_data(data_train, label_train)
X_test, y_test, y_test_dict, le_test = _transform_data(data_test, label_test)

#train model KNN, k =3
model = KNeighborsClassifier(n_neighbors = 3)
model.fit(X_train, y_train)

KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=None, n_neighbors=3, p=2,
           weights='uniform')

y_train_pred = model.predict(X_train)

uniq_labels = list(y_train_dict.values())
print(classification_report(y_train, y_train_pred, target_names = uniq_labels))



