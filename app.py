from flask import Flask, request
from flask_cors import CORS
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
import pickle
import os
import numpy as np
import pandas as pd
from flask import jsonify
from PIL import Image
from keras.models import load_model
from keras.preprocessing import image
from io import BytesIO
import base64
import json
import re
import sys
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from flask import Flask, request, jsonify

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1000 * 1000

mnist_model = None

# Emotionsmodell.h5
def load_emotion_model():
    global emotion_model
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "emotionmodell.h5")
    emotion_model = load_model(model_path)

# model1.h5
def load_model1_h5():
    global emotion_model1
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model1.h5")
    emotion_model1 = load_model(model_path)

# 'tensorflow => zweiter Versuch für modell1.h5
def load_modell1h5_model():
    global modell1h5
    modell1h5 = tf.keras.models.load_model('model1.h5')

# Emotionsmodel3_backup
def load_emotion_model2():
    global emotion_model
    model_path = "emotionmodel3_backup.h5"
    emotion_model = load_model(model_path)

#-----------------------------------------------------------------------------#

# Emotionsmodell3_backup

# Laden des gespeicherten Modells und der Emotionslabels
model = tf.keras.models.load_model('hfmodel_versuch_vier/mobilenetv3-imagenet')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    img = image.load_img(file, target_size=(48, 48), color_mode='grayscale')
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    
    # Emotionserkennung
    prediction = model.predict(img)
    predicted_label = emotion_labels[np.argmax(prediction)]

    return jsonify({'emotion': predicted_label})

#-----------------------------------------------------------------------------#
    
# Test  => Deebface- Anwendung
def perform_emotion_recognition():
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Kamera öffnen
    cap = cv2.VideoCapture(0)

    # Überprüfen, ob die Kamera erfolgreich geöffnet wurde
    if not cap.isOpened():
        raise IOError("Kann die Kamera nicht öffnen")

    # Erfassen eines Einzelbildes
    ret, frame = cap.read()

    # Überprüfen, ob das Einzelbild erfolgreich erfasst wurde
    if not ret:
        raise IOError("Fehler beim Erfassen des Bildes von der Kamera")

    # Freigeben der Kamera
    cap.release()

    # Laden des Haar-Cascade-Klassifikators
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Konvertieren in Graustufen
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Gesichtserkennung
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    # Emotionserkennung (imports in pyhtonanywhere nicht vergessen!)
    predictions = DeepFace.analyze(frame)

    # Extrahieren der dominanten Emotion aus der Liste 'predictions'
    dominant_emotion = predictions[0]['dominant_emotion']

    # Zeichnen des Rechtecks um die Gesichter und Anzeigen der Emotion
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame,
                    dominant_emotion,
                    (x, y - 10),  # Position des Textes über dem Rechteck
                    font, 1,
                    (0, 255, 0),
                    2,
                    cv2.LINE_4)

    # Anzeigen des Bildes mit Rechteck und Emotion
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

    return dominant_emotion

#-----------------------------------------------------------------------------#

# Das trainierte Emotion-Modell laden (z.B. emotionmodell.h5)
model = tf.keras.models.load_model('emotionmodell.h5')

#  Route für die Emotionsvorhersage definieren
@app.route('/api/predict-emotion', methods=['POST'])
def predict_emotion():
    try:
        image_data = re.sub('^data:image/.+;base64,', '', request.get_json()['image'])
        im = Image.open(BytesIO(base64.b64decode(image_data)))
        im.save('emotion_image.png')
        processed_image = preprocess_image(im)  # Funktion zur Bildverarbeitung anpassen

        # Emotionsvorhersage durchführen
        predicted_emotion = model.predict(processed_image)

        # Emotionskategorie auswählen
        emotion_category = get_emotion_category(predicted_emotion)

        # Emotionsvorhersage als JSON zurückgeben
        return json.dumps({'emotion': emotion_category}), 200, {'ContentType': 'application/json'}
    except Exception as err:
        return json.dumps({'error': str(err)}), 500, {'ContentType': 'application/json'}

def preprocess_image(image):
    # Bildverarbeitungsschritte anpassen, z.B. Skalierung, Konvertierung usw.
    processed_image = image.resize((160, 160)).convert('RGB')
    processed_image = np.asarray(processed_image)
    processed_image = np.reshape(processed_image, ((1, 160, 160, 3)))
    return processed_image

def get_emotion_category(emotion_prediction):
    #  Logik für die Auswahl der Emotionskategorie basierend auf den Vorhersagewerten anpassen
    class_names = ['Anger', 'Disgust', 'Fear', 'Happiness', 'neutral', 'Sadness', 'Surprise']
    predicted_class_index = np.argmax(emotion_prediction)
    emotion_category = class_names[predicted_class_index]
    return emotion_category

#-----------------------------------------------------------------------------#
"""
# Beispiel für Umsetzung ohne Hugging Face
pythonanywhere free can't load tensorflow because the package is to large.
if you want to use it localy, remove the comments 'tensorflow-local' in this file
@app.route('/api/prediction/cats', methods=['POST'])
def predict_cats():
    try:
        image_data = re.sub('^data:image/.+;base64,', '', request.get_json()['image'])
        im = Image.open(BytesIO(base64.b64decode(image_data)))
        im.save('cat.png')
        cat = im.resize((160, 160)).convert('RGB')
         #singel_tmp_image = tf.keras.utils.load_img('leopard.jpg',
                                            #target_size=IMG_SIZE,
                                            #interpolation="nearest")
        # image to numpy
        np_image = np.asarray(cat)
        np_image = np.reshape(np.asarray(np_image), ((1, 160, 160, 3)))
        predicted_cats = cats_model.predict([np_image])
        class_names = ['cheetah', 'leopard', 'lion', 'tiger']
        print(predicted_cats)

        return json.dumps({'prediction': list(zip(class_names, predicted_cats[0].astype(float)))}), 200, {'ContentType': 'application/json'}
    except Exception as err:
        return json.dumps({'error': str(err)})
"""

#-----------------------------------------------------------------------------#

@app.route("/")
def hello_world():

    print(request.args)
    return "<p>Current model!</p>" + str(mnist_model)

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(("* Loading model and Flask starting server..."
        "please wait until server has fully started"))
    load_emotion_model()
    load_model1_h5()
    load_emotion_model2()

    print(sys.executable)
    print('running')
    app.run()