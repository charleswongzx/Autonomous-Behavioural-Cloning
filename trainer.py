import csv
import os

from keras.models import Sequential
from keras.layers import *

from sklearn.model_selection import train_test_split

from generator import generate

# Parameters
gen_batch_size = 32
dropout = 0.5
epochs = 2

# List folders containing training data
dirs = []
for item in os.listdir('sim_data'):
    if os.path.isdir(os.path.join('sim_data', item)):
        dirs.append(item)

# Bringing in the driving log csv file(s) from respective dirs
lines = []
for i in dirs:
    csvfile =  open('sim_data/'+i+'/driving_log.csv')
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

# Splitting train/validation sets
X_train, X_valid = train_test_split(lines, test_size=0.2)

# Creating generators for each dataset
X_train_gen = generate(X_train, gen_batch_size)
X_valid_gen = generate(X_valid, gen_batch_size)

model = Sequential()

# Cropping and normalisation
model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: x / 255.0 - 0.5))

# NVIDIA End-to-end neural net implementation
model.add(Convolution2D(filters=24,kernel_size=5, strides=(2,2), activation='relu'))
model.add(Convolution2D(filters=36,kernel_size=5,strides=(2,2),activation='relu'))
model.add(Convolution2D(filters=48,kernel_size=5,strides=(2,2),activation='relu'))
model.add(Convolution2D(filters=64,kernel_size=3,activation='relu'))
model.add(Convolution2D(filters=64,kernel_size=3,activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(dropout))
model.add(Dense(50))
model.add(Dropout(dropout))
model.add(Dense(10))
model.add(Dense(1, activation='linear'))

print('Number of samples: ', len(lines)*2)

# Loss function and optimiser step
model.compile(loss='mse', optimizer='adam')
model.fit_generator(X_train_gen, steps_per_epoch=len(X_train)/gen_batch_size, validation_data=X_valid_gen, validation_steps=len(X_valid)/gen_batch_size, epochs=epochs)
print('Training complete!')
model.save('nvidia_29042017_ucy_3cams.h5')
print('Model saved!')