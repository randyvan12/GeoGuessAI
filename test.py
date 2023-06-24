import tensorflow as tf
import numpy as np
from PIL import Image

# Load the model
model = tf.keras.models.load_model('complete_saved_model/')

# List of image paths
image_paths = ['C:/Users/randy/Desktop/archive/Chicago1.jpg', 
               'C:/Users/randy/Desktop/archive/Chicago2.jpg', 
               'C:/Users/randy/Desktop/archive/Chicago3.jpg', 
               'C:/Users/randy/Desktop/archive/Detroit1.jpg',
               'C:/Users/randy/Desktop/archive/Detroit2.jpg', 
               'C:/Users/randy/Desktop/archive/Detroit3.jpg',
               'C:/Users/randy/Desktop/archive/New York1.jpg',
               'C:/Users/randy/Desktop/archive/New York2.jpg', 
               'C:/Users/randy/Desktop/archive/New York3.jpg',
               'C:/Users/randy/Desktop/archive/San1.jpg',
               'C:/Users/randy/Desktop/archive/San2.jpg', 
               'C:/Users/randy/Desktop/archive/San3.jpg',
               'C:/Users/randy/Desktop/archive/Wash1.jpg',
               'C:/Users/randy/Desktop/archive/Wash2.jpg', 
               'C:/Users/randy/Desktop/archive/Wash3.jpg']

# Preprocess the images
img_height = 320
img_width = 320
images = []

for image_path in image_paths:
    image = Image.open(image_path)
    image = image.resize((img_width, img_height))
    image = np.array(image)
    image = image / 255.0  # Normalize to [0,1] (optional depending on your model preprocessing)
    images.append(image)

images = np.stack(images, axis=0)  # Stack images to create a batch

# Make predictions
predictions = model.predict(images)

# Decode prediction
class_names = ['Chicago', 'City of New York', 'Detroit', 'San Francisco', 'Washington']

for i in range(len(predictions)):
    prediction = predictions[i]
    image_path = image_paths[i]
    predicted_class_index = np.argmax(prediction)
    predicted_class_name = class_names[predicted_class_index]
    
    # Convert probabilities to percentages
    probabilities_percentage = [prob * 100 for prob in prediction]
    
    print(f'Image {i + 1}:')
    print(f'   Image Path: {image_path}')
    print(f'   Predicted Class: {predicted_class_name}')
    
    # Print each class probability as percentage
    print('   Probabilities:')
    for j, prob in enumerate(probabilities_percentage):
        print(f'      {class_names[j]}: {prob:.2f}%')