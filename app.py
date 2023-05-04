import collections
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, render_template, request
from keras.models import model_from_json

app = Flask(__name__)

upload_folder = 'uploaded'
meta_folder = 'meta'


@app.route('/')
def upload_file():
    return render_template('upload.html')


@app.route('/result', methods=['GET', 'POST'])
def get_result():
    preds = None
    if request.method == 'POST':
        file = request.files['file']
        filename = file.filename
        if (filename == None) or (not filename.endswith(".csv")):
            return "Invalid file. Support only '.csv' files."
        file.save(os.path.join(upload_folder, filename))
        # .csv file saved

        preds = doProcess(filename)
        print(preds)
        return render_template('pred.html', preds=preds)

    return render_template('upload.html')


def doProcess(filename):
    df = pd.read_csv('uploaded/' + filename)
    df.drop([0])

    json_file = open('meta/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # load weights into new model
    loaded_model.load_weights('meta/weights.hdf5')
    y_pred1 = tf.argmax(loaded_model.predict(df), axis=-1)
    y_pred1 = np.array(y_pred1).tolist()
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
