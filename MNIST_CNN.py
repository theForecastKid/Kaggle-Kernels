#This kernel gets aprox 99.6 accuraccy score on Kaggle Leaderboard

import numpy as np
from subprocess import check_output
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation, Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter


train_file = "train.csv"
test_file = "test.csv"
output_file = "submission.csv"

mnist_dataset = np.loadtxt(train_file, skiprows=1, dtype='int', delimiter=',')

#First we use 0.125 to calculate the optimum number of iterations
#We also need to set patience = 3
val_split = 0.001
n_raw = mnist_dataset.shape[0]
n_val = int(n_raw * val_split + 0.5)
n_train = n_raw - n_val

np.random.shuffle(mnist_dataset)
x_val, x_train = mnist_dataset[:n_val,1:], mnist_dataset[n_val:,1:]
y_val, y_train = mnist_dataset[:n_val,0], mnist_dataset[n_val:,0]

x_train = x_train.astype("float32")/255.0
x_val = x_val.astype("float32")/255.0
y_train = np_utils.to_categorical(y_train)
y_val = np_utils.to_categorical(y_val)

n_classes = y_train.shape[1]
x_train = x_train.reshape(n_train, 28, 28, 1)
x_val = x_val.reshape(n_val, 28, 28, 1)

model = Sequential()

model.add(Conv2D(32, 3, 3, input_shape = (28, 28, 1)))
model.add(Activation('relu'))
model.add(Conv2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, 3, 3))
model.add(Activation('relu'))
model.add(Conv2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.50))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.50))
model.add(Dense(10))
model.add(Activation('softmax'))

#Kera's ImageDataGenerator doesn't support elastic transformations, 
#we need to implement it first

def elastic_transform(image, alpha, sigma, random_state=None):
  
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))

    distored_image = map_coordinates(image, indices, order=1, mode='reflect')
    return distored_image.reshape(image.shape)

#This preprocess funtion apply elastic transformation and adds Gaussian noise

def preprocess(img):
    img = elastic_transform(img.reshape(28, 28), np.random.randint(4), 1)
    img = img + np.abs(np.random.normal(0.1, 0.3))*np.random.rand(28,28) #Gaussian noise
    return img.reshape(28, 28, 1)

datagen = ImageDataGenerator(zoom_range = [0.9, 1.1],
                            height_shift_range = 0.1,
                            width_shift_range = 0.1,
                            rotation_range = 10,
                            shear_range = 0.2,
                            preprocessing_function = preprocess)

model.compile(loss='categorical_crossentropy', optimizer = Adam(lr=1e-4), metrics = ["accuracy"])

callbacks = [EarlyStopping(monitor='val_loss', min_delta=0, patience=60, verbose=2, mode='auto'),
            ModelCheckpoint('mnist.h5', monitor='val_loss', save_best_only=True, verbose=0)]

hist = model.fit_generator(datagen.flow(x_train, y_train, batch_size = 32),
                           samples_per_epoch = n_train, 
                           nb_epoch = 60, 
                           validation_data = (x_val, y_val),
                           callbacks = callbacks)


mnist_testset = np.loadtxt(test_file, skiprows=1, dtype='int', delimiter=',')
x_test = mnist_testset.astype("float32")/255.0
n_test = x_test.shape[0]
x_test = x_test.reshape(n_test, 28, 28, 1)

y_test = model.predict(x_test, batch_size=32)

y_index = np.argmax(y_test,axis=1)

with open(output_file, 'w') as f :
    f.write('ImageId,Label\n')
    for i in range(0,n_test) :
        f.write("".join([str(i+1),',',str(y_index[i]),'\n']))
