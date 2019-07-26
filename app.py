from flask import Flask, render_template, request, jsonify
import numpy as np
from pandas import *
import tensorflow as tf
import cv2
import keras.models
#for regular expressions, saves time dealing with string data
import re
import base64, json
import io
from io import BytesIO
from PIL import Image
from PIL import Image, ImageChops, ImageDraw
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator, img_to_array

#system level operations (like loading files)
import sys 
#for reading operating system data
import os

app = Flask(__name__)

def convertImage(imgData1):
    imgstr = re.search(b'base64,(.*)',imgData1).group(1)
    with open('output.png','wb') as output:
        output.write(base64.b64decode(imgstr))


# centering input digit
def __centering_img(img):

    width, height = img.size
    left,top,right,bottom = width, height, -1, -1
    imgpix = img.getdata()

    for y in range(height):
        yoffset = y*width
        for x in range(width):
            if imgpix[yoffset + x] < 255:

                # do not use GetPixel and SetPixel, it is so slow.
                if x < left: left   = x
                if y < top: top     = y
                if x > right: right = x
                if y > bottom: bottom = y   

    shiftX = int((left + (right - left) / 2) - width / 2)
    shiftY = int((top + (bottom - top) / 2) - height / 2)

    return ImageChops.offset(img, -shiftX, -shiftY)


        

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/DigitRecognition',methods=['GET','POST'])
def predict():
    retJson = {"predict_digit" :"Err", "detect_img" :"", "centering_img" :"", "prob" :{}}
    if request.method == 'POST':
        # request.body
        imgpath = BytesIO(base64.urlsafe_b64decode(request.form['img']))
        
        img = Image.open(imgpath).convert('L')
        # centering input digit
        img = __centering_img(img)

        img.thumbnail((28, 28))
        img = np.array(img, dtype=np.float32)
        img = 1 - np.array(img / 255)
        img = img.reshape(28, 28)
        img = np.expand_dims(img, axis=0)
        
        #res =  cnn.predict(postImg)
        new_model = tf.keras.models.load_model('my-model.model')
        res = new_model.predict(img)
    
        if res is not None:
            retJson["predict_digit"] = str(np.argmax(res))

            """for i, item in enumerate(res):
                retJson["prob"][i] = float(item*100)"""
        
            # save digits 
            """ I do not want to save the image
            postImg = Image.open(imgpath)
            postImg.save("./{}_{}.png".format(datetime.now().strftime('%m-%d_%H.%M.%S'),retJson["predict_digit"] ))
            """

    return json.dumps(retJson)



if (__name__ == "__main__"):
    app.debug = True
    app.run()