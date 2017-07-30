# The CNN model will be used in this project
# Credits to NVIDIA: https://arxiv.org/pdf/1604.07316.pdf

# Coded by: Musa Atlıhan (github.com/wphw)
# 2017-07-29 (github.com/wphw/CarND-Behavioral-Cloning-P3)


# parameters
csv_path = 'data/driving_log.csv'
save_name = 'weights-best.hdf5'
batch_size = 32
epochs = 10


# load samples
import csv

samples = []
with open(csv_path) as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        samples.append(row)


# split samples
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

train_samples, valid_samples = train_test_split(samples, test_size=0.3, random_state=2)


import cv2
import numpy as np

# generator for to load the batch
def generator(samples, batch_size=32):
    n_samples = len(samples)
    while 1:
        for offset in range(0, n_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                path = './data/IMG/' + batch_sample[0].split('/')[-1]
                center_image = cv2.imread(path)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
            X = np.array(images)
            y = np.array(angles)
            yield shuffle(X, y)

# define generators
train_generator = generator(train_samples, batch_size=batch_size)
valid_generator = generator(valid_samples, batch_size=batch_size)


# create model
import keras
from keras.models import Sequential
from keras.layers import Cropping2D, Lambda
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten, Dense, Dropout
from keras.callbacks import ModelCheckpoint

# the model
model = Sequential()

model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(160, 320, 3)))
model.add(Lambda(lambda x: x / 127.5 - 1.))
model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='relu'))
model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))
model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

# callbacks
checkpoint = ModelCheckpoint(save_name, monitor='val_acc', verbose=1,
                                                    save_best_only=True, mode='max')
callbacks_list = [checkpoint]

model.fit_generator(train_generator, steps_per_epoch=len(train_samples)//batch_size,
                    validation_data=valid_generator, validation_steps=\
                    len(valid_samples)//batch_size, callbacks=callbacks_list,
                    epochs=epochs)

model.save(save_name)