# The CNN model will be used in this project
# Credits to NVIDIA: https://arxiv.org/pdf/1604.07316.pdf

# Coded by: Musa Atlıhan (github.com/wphw)
# 2017-07-29 (github.com/wphw/CarND-Behavioral-Cloning-P3)


from utils import *
import argparse
import os
import csv
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Cropping2D, Lambda
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten, Dense, Dropout
from keras.callbacks import ModelCheckpoint


# generator for to load the batch
def generator(samples, batch_size=32):
    n_samples = samples.shape[0]
    while 1:
        for offset in range(0, n_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            correction = 0.2
            for batch_sample in batch_samples:
                left_path = './' + csv_path + 'IMG/' + batch_sample[1].split('/')[-1]
                center_path = './' + csv_path + 'IMG/' + batch_sample[0].split('/')[-1]
                right_path = './' + csv_path + 'IMG/' + batch_sample[2].split('/')[-1]

                left_image = cv2.imread(left_path)
                center_image = cv2.imread(center_path)
                right_image = cv2.imread(right_path)

                flipped_left_image = cv2.flip(left_image, 1)
                flipped_center_image = cv2.flip(center_image, 1)
                flipped_right_image = cv2.flip(right_image, 1)
                
                center_angle = float(batch_sample[3])
                left_angle = center_angle + correction
                right_angle = center_angle - correction
                
                images.extend([left_image, center_image, right_image, 
                    flipped_left_image, flipped_center_image, flipped_right_image])
                #images.extend([left_image, center_image, right_image])
                angles.extend([left_angle, center_angle, right_angle,
                    -left_angle, -center_angle, -right_angle])
                #angles.extend([left_angle, center_angle, right_angle])
            X = np.array(images)
            y = np.array(angles)
            yield shuffle(X, y)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Model Args')
    parser.add_argument(
        'pct',
        type=float,
        nargs='?',
        default=0,
        help='Percentage (float) to remove data with the angle=0. Example: (float)100 removes all.'
    )
    args = parser.parse_args()
    
    # parameters
    csv_path = 'data-joystick-5-loops/'
    csv_name = 'driving_log.csv'
    
    monitor = 'val_loss'
    save_dir = 'saved-5-loops-ptc-' + str(args.pct) + '/'
    save_name = 'weights-{epoch:03d}-val_loss-{val_loss:.5f}.hdf5'
    
    batch_size = 32
    epochs = 5
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # load samples
    samples = []
    with open(csv_path + csv_name) as f:
        reader = csv.reader(f)
        #next(reader)
        for row in reader:
            samples.append(row)
    
    
    # split samples    
    train_samples, valid_samples = train_test_split(samples, test_size=0.3, random_state=100)
    
    train_samples = np.array(train_samples)
    print(train_samples.shape)
    # reduce the number of rows with angle=0
    # pct is the percentage to remove elements, example: (int)100 removes all
    train_samples = exclude_by_value(train_samples, 3, '0', pct=args.pct)
    print(train_samples.shape)
    valid_samples = np.array(valid_samples)

    # define generators
    train_generator = generator(train_samples, batch_size=batch_size)
    valid_generator = generator(valid_samples, batch_size=batch_size)
    
    
    # create model
    model = Sequential()
    
    model.add(Cropping2D(cropping=((60, 20), (0, 0)), input_shape=(160, 320, 3)))
    model.add(Lambda(lambda x: (x / 127.5) - 1.))
    model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    
    #sgd = keras.optimizers.SGD(lr=0.01)
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    
    # callbacks
    checkpoint = ModelCheckpoint(
        save_dir + save_name, monitor=monitor, verbose=1, save_best_only=False, mode='min')
    callbacks_list = [checkpoint]
    
    model.fit_generator(
        train_generator,
        steps_per_epoch=(len(train_samples)*6)//batch_size,
        validation_data=valid_generator,
        validation_steps=(len(valid_samples)*6)//batch_size,
        callbacks=callbacks_list,
        epochs=epochs)