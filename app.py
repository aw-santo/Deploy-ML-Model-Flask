# import os
# from flask import Flask
# app = Flask(__name__)
#
# @app.route("/")
# def main():
#     return "Welcome!"
#
# @app.route('/hru')
# def hello():
#     return 'I am good, how about you?'
#
# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=8880)

from flask import Flask, render_template, request
# from werkzeug import secure_filename
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
from keras.models import Sequential, model_from_json
import os
import collections

# model = tf.keras.models.load_model('model')
app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'uploaded'


@app.route('/')
def upload_f():
    return render_template('upload.html')


# def finds():
# 	test_datagen = ImageDataGenerator(rescale = 1./255)
# 	vals = ['Cat', 'Dog'] # change this according to what you've trained your model to do
# 	test_dir = 'uploaded'
# 	test_generator = test_datagen.flow_from_directory(
# 			test_dir,
# 			target_size =(224, 224),
# 			color_mode ="rgb",
# 			shuffle = False,
# 			class_mode ='categorical',
# 			batch_size = 1)
#
# 	pred = model.predict_generator(test_generator)
# 	print(pred)
# 	return str(vals[np.argmax(pred)])

@app.route('/uploader', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        filename = file.filename
        if filename==None or not filename.endswith(".csv"):
            return "Invalid file. Support only '.csv' files."
        file.save(file.filename)
    # .csv file saved

    preds =  doProcess(filename)
    # op = preds.tobytes()
    print(preds)
    return preds
    # return "Test"

def doProcess(filename):
    df = pd.read_csv(filename)
    df.drop([0])
    print(df.head())
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("weights.hdf5")
    print("Loaded model from disk")
    print(df.shape)
    y_pred1 = tf.argmax(loaded_model.predict(df), axis=-1)
    y_pred1 = np.array(y_pred1).tolist()
    print(y_pred1)
    res = maxOccurences(y_pred1)
    return info[res]

def maxOccurences(y_preds):
    counter = collections.Counter(y_preds)
    max_occurrence = max(counter.values())
    for number, count in counter.items():
        if count == max_occurrence:
            return number

info = {
        0: "Normal beat",
        1: " Left bundle branch block beat",
        2: "Right bundle branch block beat",
        3: "Atrial Premature Beat",
        4: "Premature Ventricular Beat"
        }

if __name__ == '__main__':
    app.run()
