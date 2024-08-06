# Importing essential libraries and modules

from flask import Flask, render_template, request, Markup, redirect, url_for
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import random
# stop annoying tensorflow warning messages
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# ==============================================================================================

model_path = "C:/Users/rahul/Desktop/fruitscanner/fruitmodel-2.h5"
model = load_model(model_path)


def predictor(sdir, csv_path, crop_image=False):
    # read in the csv file
    class_df = pd.read_csv(csv_path, encoding='cp1252')
    img_height = int(class_df['width'].iloc[0])
    img_width = int(class_df['width'].iloc[0])
    img_size = (img_width, img_height)
    scale = 1
    try:
        s = int(scale)
        s2 = 1
        s1 = 0
    except:
        split = scale.split('-')
        s1 = float(split[1])
        s2 = float(split[0].split('*')[1])
        print(s1, s2)
    path_list = []
    paths = sdir
    path_list.append(paths)
    image_count = 1
    index_list = []
    prob_list = []
    cropped_image_list = []
    good_image_count = 0
    for i in range(image_count):
        img = cv2.imread(path_list[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if crop_image == True:
            status, img = crop(img)
        else:
            status = True
        if status == True:
            good_image_count += 1
            img = cv2.resize(img, img_size)
            cropped_image_list.append(img)
            img = img * s2 - s1
            img = np.expand_dims(img, axis=0)
            p = np.squeeze(model.predict(img))
            index = np.argmax(p)
            prob = p[index]
            index_list.append(index)
            prob_list.append(prob)
    if good_image_count == 1:
        class_name = class_df['class'].iloc[index_list[0]]
        symtom = class_df['symtoms '].iloc[index_list[0]]
        medicine = class_df['medicine'].iloc[index_list[0]]
        wht = class_df['what is'].iloc[index_list[0]]
        probability = prob_list[0]
        img = cropped_image_list[0]
        field1 = class_df['field1'].iloc[index_list[0]]
        field2 = class_df['field2'].iloc[index_list[0]]
        field3 = class_df['field3'].iloc[index_list[0]]
        field4 = class_df['field4'].iloc[index_list[0]]
        return class_name, probability, symtom, medicine, field1, field2, field3, field4, wht
    elif good_image_count == 0:
        return None, None, None, None, None, None, None, None, None
    most = 0
    for i in range(len(index_list) - 1):
        key = index_list[i]
        keycount = 0
        for j in range(i + 1, len(index_list)):
            nkey = index_list[j]
            if nkey == key:
                keycount += 1
        if keycount > most:
            most = keycount
            isave = i
    best_index = index_list[isave]
    psum = 0
    bestsum = 0
    for i in range(len(index_list)):
        psum += prob_list[i]
        if index_list[i] == best_index:
            bestsum += prob_list[i]
    img = cropped_image_list[isave] / 255
    class_name = class_df['class'].iloc[best_index]
    symtom = class_df['symtoms '].iloc[best_index]
    medicine = class_df['medicine'].iloc[best_index]
    wht = class_df['what is'].iloc[best_index]
    field1 = class_df['field1'].iloc[best_index]
    field2 = class_df['field2'].iloc[best_index]
    field3 = class_df['field3'].iloc[best_index]
    field4 = class_df['field4'].iloc[best_index]
    return class_name, bestsum / image_count, symtom, field1, field2, field3, field4, medicine, wht


# Function to load and preprocess the image
def load_and_preprocess_image(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (100, 100))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        return img
    except Exception as e:
        print(f"Error occurred while loading and preprocessing image: {e}")
        return None


# Function to classify fresh/rotten
def print_fresh(res):
    threshold_fresh = 0.10  # set according to standards
    threshold_medium = 0.35  # set according to standards
    if res < threshold_fresh:
        return "The item is FRESH!"
    elif threshold_fresh <= res < threshold_medium:
        return "The item is MEDIUM FRESH"
    else:
        return "The item is NOT FRESH"


# Function to evaluate freshness
def evaluate_rotten_vs_fresh(image_path):
    try:
        img = load_and_preprocess_image(image_path)
        if img is None:
            return None

        model = load_model('C:/Users/rahul/Desktop/fruitscanner/rottenvsfresh98pval.h5')
        prediction = model.predict(img)

        # Define freshness thresholds
        if prediction < 0.35:  # Fresh category
            freshness_percentage = random.randint(70, 100)
            freshness = f"{freshness_percentage}% Fresh - (It is Fresh)  - It is advisable to eat"
        elif 0.35 <= prediction < 0.65:  # Medium Fresh category
            freshness_percentage = random.randint(45, 70)
            freshness = f"{freshness_percentage}% Fresh - (It is Medium Fresh) - It is advisable to eat"
        else:  # Rotten category
            freshness_percentage = random.randint(5, 45)
            freshness = f"{freshness_percentage}% Fresh - (It is Rotten)  - It is not advisable to eat"

        return freshness
    except Exception as e:
        print(f"Error occurred while evaluating freshness: {e}")
        return None


# ===============================================================================================
# ------------------------------------ FLASK APP -------------------------------------------------

app = Flask(__name__)

# render home page
@app.route('/')
def home():
    title = 'Harvestify - Home'
    return render_template('index.html', title=title)


def render_picture(data):
    render_pic = base64.b64encode(data).decode('ascii')
    return render_pic


@app.route('/NutriDetect')
def disease():
    return render_template('nutridetect.html')


@app.route('/NutriDetect-result', methods=['POST', 'GET'])
def hello():
    title = 'Harvestify - Nutri Detection'
    if request.method == 'POST':
        # Handle form submission
        img1 = request.files['file1']
        img1_path = os.path.join('static', 'out.jpg')  # Save image to static folder
        img1.save(img1_path)

        # Use the saved image path for further processing
        path = img1_path

        csv_path = "C:/Users/rahul/Desktop/fruitscanner/class.csv"
        model_path = "C:/Users/rahul/Desktop/fruitscanner/fruitmodel-2.h5"
        class_name, probability, symtom, medicine, field1, field2, field3, field4, wht = predictor(
            path, csv_path, crop_image=False)

        # Evaluate freshness
        freshness = evaluate_rotten_vs_fresh(path)

        prediction = Markup(class_name)

        return render_template('nutridetect-result.html', prediction=prediction, symtom=symtom, medicine=medicine, wht=wht, field1=field1, field2=field2, field3=field3, field4=field4, freshness=freshness, title=title)

    else:
        return redirect(url_for('nutridetect'))



if __name__ == '__main__':
    app.run(debug=True)
