import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import PIL
import matplotlib.pyplot as plt

# https://www.youtube.com/watch?v=q7ZuZ8ZOErE
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
# model = Sequential([
#   layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
#   layers.Conv2D(16, 3, padding='same', activation='relu'),
#   layers.MaxPooling2D(),
#   layers.Conv2D(32, 3, padding='same', activation='relu'),
#   layers.MaxPooling2D(),
#   layers.Conv2D(64, 3, padding='same', activation='relu'),
#   layers.MaxPooling2D(),
#   layers.Conv2D(128, 3, padding='same', activation='relu'),
#   layers.MaxPooling2D(),
#   layers.Flatten(),
#   layers.Dense(256, activation='relu'),
#   layers.Dense(128, activation='relu'),
#   layers.Dense(64, activation='relu'),
#   layers.Dense(32, activation='relu'),
#   layers.Dense(16, activation='relu'),
#   layers.Dense(num_classes)
# ])
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

# AUTOTUNE = tf.data.AUTOTUNE

# train_ds = train.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
# val_ds = validation.cache().prefetch(buffer_size=AUTOTUNE)

# normalization_layer = layers.Rescaling(1./255)
# normalized_ds = train.map(lambda x, y: (normalization_layer(x), y))
# image_batch, labels_batch = next(iter(normalized_ds))
# first_image = image_batch[0]
# Notice the pixel values are now in `[0,1]`.
# print(np.min(first_image), np.max(first_image))

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
# def augment(x, y):
#     return 0
# train = train.map(augment)
# model = tf.keras.models.Sequential([
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dense(64, activation='relu'),
#     tf.keras.layers.Dense(10, activation='softmax')
# ])
# # load mnist
# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# # reshape, normalize, and convert inputs to float32
# x_train = (x_train / 255.).reshape([-1, 784]).astype(np.float32)
# x_test = (x_test / 255.).reshape([-1, 784]).astype(np.float32)

# # convert labels to one-hot vectors
# y_train, y_test = tf.one_hot(y_train, 10), tf.one_hot(y_test, 10)

# # prepare for training
# train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
# train_data = train_data.shuffle(500).batch(32)

# model.compile(optimizer='adam', loss='categorical_crossentropy')
# model.fit(train_data, epochs=10)

# def accuracy(y_pred, y_true):
#     correct_prediction = tf.equal(tf.argmax(y_pred, -1), tf.argmax(y_true, -1))
#     return tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis=-1)

# pred = model(x_test)
# print(f'Test accuracy: {accuracy(pred, y_test)}')
# print(tf.version)