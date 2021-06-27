import os
from flask import Flask, render_template, request,Response, jsonify
from flask_dropzone import Dropzone
import cv2
import base64
import glob
from mtcnn.mtcnn import MTCNN
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from PIL import Image
from keras import layers
from keras import callbacks
from PIL import Image
import numpy as np
def detectMask(filename):
    model = keras.Sequential([
        layers.Conv2D(100, (3,3), activation='relu', input_shape=(50,50,3)),
        layers.MaxPooling2D(2,2),
        
        layers.Conv2D(100, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(50, activation='relu'),
        layers.Dense(2, activation='softmax')
    ])

    model.load_weights("MaskDetectorInspector/model_weights.h5")

    pixels = plt.imread(filename)
    detector = MTCNN()

    row, col, dep = pixels.shape
    isRGB = True
    if dep>3:
        isRGB = False
        rgb = np.zeros((row, col, 3),dtype='float32')
        r, g, b, a = pixels[:,:,0], pixels[:,:,1], pixels[:,:,2], pixels[:,:,3]

        a = np.asarray(a,dtype='float32' )/255.0
        R, G, B = (255,255,255)

        rgb[:,:,0] = r
        rgb[:,:,1] = g
        rgb[:,:,2] = b

        pixels2 = np.asarray(rgb, dtype='uint8')
        pixels = np.copy(pixels2)

    faces = detector.detect_faces(pixels)

    output = []
    for face in faces:
        x,y,w,h = face["box"]

        imgVals = []
        for i in range(h):
            row = []
            for j in range(w):
                px = x+j
                py = y+i
                row.append(pixels[py][px])
                imgVals.append(row)
        
        imgVals = np.array(imgVals)
        image = Image.fromarray(imgVals)
        image = image.resize((50,50))
        imgVals = np.array(image)

        result = model.predict(np.array([imgVals,]))
        if np.argmax(result) == 0:
            output.append(1)
        else:
            output.append(0)
    
    return output
basedir = os.path.abspath(os.path.dirname(__file__))


app = Flask(__name__)
app.config.update(
    UPLOADED_PATH= os.path.join(basedir,'uploads'),
    DROPZONE_MAX_FILE_SIZE = 1024,
    DROPZONE_TIMEOUT = 5*60*1000)

@app.route("/")
def index():
    return render_template("index.html")
dropzone = Dropzone(app)

@app.route('/upload.html/',methods=['POST','GET'])
def upload():
    if request.method == 'POST':
        f = request.files.get('file')
        f.save(os.path.join(app.config['UPLOADED_PATH'],f.filename))
        folder_path = r'c:\Users\Manhar\Desktop\Flask\uploads'
        file_type = '\*jpg'
        files = glob.glob(folder_path + file_type)
        max_file = max(files, key=os.path.getctime)
        results = detectMask(max_file)
        return render_template('upload.html', values=[len(results),sum(results)])
    return render_template('upload.html')
    





if __name__ == "__main__":
    app.run(debug=True)
