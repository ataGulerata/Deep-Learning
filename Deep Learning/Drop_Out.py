import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
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

print("Images shape:", images.shape)
print("Labels shape:", labels.shape)


split_ratio = 0.8  
split_index = int(len(images) * split_ratio)

train_images, val_images = images[:split_index], images[split_index:]
train_labels, val_labels = labels[:split_index], labels[split_index:]

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

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=50, validation_data=(val_images, val_labels))

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
