import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import regularizers

# Change this line to import EfficientNetV2
from tensorflow.keras.applications import EfficientNetV2  # Import EfficientNetV2

file_location = 'C:/Users/randy/Desktop/archive'
img_height = 320  # Reduced image size
img_width = 320   # Reduced image size
batch_size = 32
num_classes = 5

# Load Data

# ... Your data loading code here ...

# Data augmentation
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.3),
    tf.keras.layers.experimental.preprocessing.RandomZoom(0.3),
    tf.keras.layers.experimental.preprocessing.RandomContrast(0.3),
])

# Change this line to use EfficientNetV2
base_model = EfficientNetV2(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))  # Use EfficientNetV2

# Unfreeze some layers
for layer in base_model.layers[-30:]:
    layer.trainable = True

# Build the model
inputs = tf.keras.Input(shape=(img_height, img_width, 3))
x = data_augmentation(inputs)
x = base_model(x, training=False)
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
x = Dropout(0.5)(x)
x = BatchNormalization()(x)
predictions = Dense(num_classes, activation='softmax')(x)

# Create the final model
model = Model(inputs=inputs, outputs=predictions)

# Compile the model
optimizer = Adam(learning_rate=0.00005)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.00001)

# Training
epochs = 20
history = model.fit(
    train,
    validation_data=validation,
    epochs=epochs,
    callbacks=[early_stopping, reduce_lr]
)

# Save the model
model.save('complete_saved_model/')