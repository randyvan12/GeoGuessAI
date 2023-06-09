import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import PIL
import matplotlib.pyplot as plt

file_location = 'C:/Users/randy/Desktop/archive'
img_height = 640
img_width = 640
batch_size = 32
num_classes = 5

data_augmentation = keras.Sequential([
    layers.experimental.preprocessing.RandomFlip("horizontal"),
    layers.experimental.preprocessing.RandomRotation(0.1),
    layers.experimental.preprocessing.RandomZoom(0.1),
])

model = Sequential([
  data_augmentation,
  layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2),
  layers.Flatten(),
  layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
  layers.Dense(num_classes)
])

train = tf.keras.preprocessing.image_dataset_from_directory(
    file_location,
    labels='inferred',
    label_mode = 'int',
    #class_names
    color_mode='rgb',
    batch_size = batch_size,
    image_size=(img_height, img_width),
    # shuffle = True,
    seed = 123,
    validation_split = 0.1,
    subset="training"
)
validation = tf.keras.preprocessing.image_dataset_from_directory(
    file_location,
    labels = 'inferred',
    label_mode = 'int',
    #class_names
    color_mode = 'rgb',
    batch_size = batch_size,
    image_size = (img_height, img_width),
    # shuffle = True,
    seed = 123,
    validation_split = 0.1,
    subset = "validation"
)
class_names = train.class_names
print(class_names)

# model = keras.models.load_model('complete_saved_model/')
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

epochs=10
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
history = model.fit(
  train,
  validation_data=validation,
  epochs=epochs,
  callbacks=[early_stopping]
)

model.save('complete_saved_model/')
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()