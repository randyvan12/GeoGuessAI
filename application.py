# img_viewer.py

import PySimpleGUI as sg
import os.path
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Load the model
model = tf.keras.models.load_model('complete_saved_model/')

# Preprocess the images
img_height = 320
img_width = 320

# List of class names
class_names = ['Chicago', 'City of New York', 'Detroit', 'San Francisco', 'Washington']

# First the window layout in 2 columns
file_list_column = [
    [
        sg.Text("Image Folder"),
        sg.In(size=(25, 1), enable_events=True, key="-FOLDER-"),
        sg.FolderBrowse(),
    ],
    [
        sg.Listbox(
            values=[], enable_events=True, size=(40, 20), key="-FILE LIST-"
        )
    ],
]

# For now will only show the name of the file that was chosen
image_viewer_column = [
    [sg.Text("Choose an image from list on left:")],
    [sg.Text(size=(40, 1), key="-TOUT-")],
    [sg.Image(key="-IMAGE-")],
    [sg.Text("Prediction:", size=(15, 1)), sg.Text("", size=(25, 1), key="-PREDICTION-")],
    [sg.Text("Probabilities:", size=(15, 1)), sg.Text("", size=(25, 1), key="-PROBABILITIES-")]
]

# Full layout
layout = [
    [
        sg.Column(file_list_column),
        sg.VSeperator(),
        sg.Column(image_viewer_column),
    ]
]

window = sg.Window("GeoGuesser", layout)

# Run the Event Loop
while True:
    event, values = window.read()
    if event == "Exit" or event == sg.WIN_CLOSED:
        break
    # Folder name was filled in, make a list of files in the folder
    if event == "-FOLDER-":
        folder = values["-FOLDER-"]
        try:
            # Get list of files in folder
            file_list = os.listdir(folder)
        except:
            file_list = []

        fnames = [
            f
            for f in file_list
            if os.path.isfile(os.path.join(folder, f))
            and f.lower().endswith((".png", ".gif", ".jpg", ".jpeg"))
        ]
        window["-FILE LIST-"].update(fnames)
    elif event == "-FILE LIST-":  # A file was chosen from the listbox
        try:
            filename = os.path.join(values["-FOLDER-"], values["-FILE LIST-"][0])
            window["-TOUT-"].update(filename)

            # Open image once and preprocess
            image = Image.open(filename)
            image = image.resize((img_width, img_height))
            
            # For display
            if filename.lower().endswith((".png")):
                window["-IMAGE-"].update(filename=filename)
            elif filename.lower().endswith((".jpg", ".jpeg")):
                bio = io.BytesIO()
                image.save(bio, format="PNG")
                window["-IMAGE-"].update(data=bio.getvalue())
            
            # Image prediction
            image = np.array(image)
            image = image / 255.0  # Normalize to [0,1]
            image = np.expand_dims(image, axis=0)  # Expand dimension to create a batch
            predictions = model.predict(image)
            predicted_class_index = np.argmax(predictions[0])
            predicted_class_name = class_names[predicted_class_index]

            # Convert probabilities to percentages
            probabilities_percentage = [prob * 100 for prob in predictions[0]]

            # Show in the GUI
            window["-PREDICTION-"].update(predicted_class_name)
            prob_text = "\n".join([f"{class_names[j]}: {prob:.2f}%" for j, prob in enumerate(probabilities_percentage)])
            window["-PROBABILITIES-"].update(prob_text)
            
            print(f'Predicted Class: {predicted_class_name}')
            print('Probabilities:')
            for j, prob in enumerate(probabilities_percentage):
                print(f'{class_names[j]}: {prob:.2f}%')

        except:
            pass

window.close()