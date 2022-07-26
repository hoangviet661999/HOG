from __future__ import print_function
from imutils.object_detection import non_max_suppression
import numpy as np
import imutils
import cv2
import glob
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from flask import Flask, render_template, request, make_response
import flask
import json
import io
from ast import walk
import sys
from unittest import result
from PIL import Image
import cv2
import torch
from flask import Flask, render_template, request, make_response
from werkzeug.exceptions import BadRequest
import os
import pandas
import json
# Khởi tạo model.
listOfKeys = []
# Khởi tạo flask app
app = Flask(__name__)  

def extract_body(imagePath):
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    image = cv2.imread(imagePath)
    image = imutils.resize(image, width = 400)
    orig = image.copy()
    
    plt.figure(figsize = (8, 6))
    # 1. Bounding box với ảnh gốc
    # Khởi tạo plot
    ax1 = plt.subplot(1, 2, 1)
    
    # Phát hiện người trong ảnh
    (rects, weights) = hog.detectMultiScale(img = image, winStride = (4, 4),
                                            padding = (8, 8), scale = 1.05)
    print('weights: ', weights)
    # Vẽ các bounding box xung quanh ảnh gốc
    for (x, y, h, w) in rects:
        cv2.rectangle(orig, (x, y), (x+w, y+h), (0, 0, 255), 2)
        rectFig = patches.Rectangle((x,y),w,h,linewidth=1,edgecolor='r',facecolor='none')
        ax1.imshow(orig)
        ax1.add_patch(rectFig)
        plt.title('Ảnh trước non max suppression')

    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    print('rects: ', rects.shape)
    # Sử dụng non max suppression để lấy ra bounding box cuối cùng với ngưỡng threshold = 0.65
    pick = non_max_suppression(rects, probs = None, overlapThresh=0.65)
    
    # 2. Bounding box với ảnh suppression
    # Khởi tạo plot
    ax2 = plt.subplot(1, 2, 2)
    # Vẽ bounding box cuối cùng trên ảnh
    for (xA, yA, xB, yB) in pick:
        w = xB-xA
        h = yB-yA
        cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)
        # Hiển thị hình ảnh
        plt.imshow(image)
        plt.title('Ảnh sau non max suppression')
        rectFig = patches.Rectangle((xA, yA),w,h,linewidth=1,edgecolor='r',facecolor='none')
        ax2.add_patch(rectFig)
        
    # Lấy thông tin ảnh
    filename = imagePath[imagePath.rfind("\\") + 1:]
    print("[INFO] {}: {} original boxes, {} after suppression".format(
        filename, len(rects), len(pick)))

    # cv2.imshow("Before NMS", orig)
    cv2.imshow("After NMS", image)
    cv2.waitKey(0)

@app.route('/', methods=['GET'])
def get():
    # in the select we will have each key of the list in option
    return render_template("index.html", len = len(listOfKeys), listOfKeys = listOfKeys)

@app.route('/', methods=['POST'])
def predict():
    file = extract_img(request)
    image = file.read()
    image = Image.open(io.BytesIO(image))
    image = np.array(image)
    cv2.imwrite('image.jpg', image)
    extract_body('image.jpg') 
    return render_template("index.html", len = len(listOfKeys), listOfKeys = listOfKeys)

# extract the image from the request
def extract_img(request):
    # checking if image uploaded is valid
    if 'file' not in request.files:
        raise BadRequest("Missing file parameter!")
        
    file = request.files['file']
    
    if file.filename == '':
        raise BadRequest("Given file is invalid")
    # print(request.files)  
    return file
if __name__ == "__main__":
	print("App run!")
	# Load model
	app.run(debug=True)