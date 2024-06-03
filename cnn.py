import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np



train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(r"C:\Users\ASUS\Downloads\Plant Disease\Train\Train",
                                                   target_size=(225, 225),
                                                   batch_size=32,
                                                   class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(r"C:\Users\ASUS\Downloads\Plant Disease\Validation\Validation",
                                                        target_size=(225, 225),
                                                        batch_size=32,
                                                        class_mode='categorical')


cnn = models.Sequential([
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(225, 225, 3)),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(4, activation='softmax')
])

cnn.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

cnn.fit(train_generator, validation_data=validation_generator, epochs=10)

cnn.save("model2.h5")