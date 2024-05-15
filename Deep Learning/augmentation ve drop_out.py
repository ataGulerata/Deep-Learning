import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler
import matplotlib.pyplot as plt


angry_dir = "angry/"  
neutral_dir = "neutral/"  

images = []
labels = []

for filename in os.listdir(angry_dir):
    if filename.endswith(".jpg"):
        img = cv2.imread(os.path.join(angry_dir, filename))
        img = cv2.resize(img, (32, 32))
        images.append(img)
        labels.append([1, 0])  

for filename in os.listdir(neutral_dir):
    if filename.endswith(".jpg"):
        img = cv2.imread(os.path.join(neutral_dir, filename))
        img = cv2.resize(img, (32, 32))
        images.append(img)
        labels.append([0, 1]) 

images = np.array(images)
labels = np.array(labels)


split_ratio = 0.8  
split_index = int(len(images) * split_ratio)

train_images, val_images = images[:split_index], images[split_index:]
train_labels, val_labels = labels[:split_index], labels[split_index:]

datagen = ImageDataGenerator(
    rotation_range=20,  
    width_shift_range=0.2,  
    height_shift_range=0.2,  
    shear_range=0.2,  
    zoom_range=0.2,  
    horizontal_flip=True,  
    fill_mode='nearest'  
)


def lr_decay(epoch):
    initial_learning_rate = 0.001
    decay_rate = 0.9
    decay_step = 1000
    return initial_learning_rate * (decay_rate ** (epoch // decay_step))


model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])


lr_scheduler = LearningRateScheduler(lr_decay)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


history = model.fit(datagen.flow(train_images, train_labels, batch_size=32),
                    steps_per_epoch=len(train_images) / 32, epochs=50,
                    validation_data=(val_images, val_labels),
                    callbacks=[lr_scheduler])


train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs = range(1, len(train_acc) + 1)

plt.figure(figsize=(10, 6))
plt.plot(epochs, train_acc, 'g', label='Training Accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.show()
