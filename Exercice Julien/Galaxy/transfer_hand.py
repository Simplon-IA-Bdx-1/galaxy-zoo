# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 11:12:13 2020

@author: Rodolphe
"""

import os, cv2, random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm    #Helps in visualization
from random import shuffle #to shuffle the images 


TRAIN_DIR = 'input/dogs-vs-cats-redux-kernels-edition/train' # file train
TEST_DIR = 'input/dogs-vs-cats-redux-kernels-edition/test' # file test
IMG_SIZE = 224

SHORT_LIST_TRAIN = os.listdir(TRAIN_DIR)[:5000] + os.listdir(TRAIN_DIR)[-5000:] #using a subset of data as resouces as limited. 
SHORT_LIST_TEST = os.listdir(TEST_DIR)

def label_img(img): 
    word_label = img.split('.')[0]
    # conversion to one-hot array [cat,dog]
    #                            [much cat, no dog]
    if word_label == 'cat': 
        print('cat')
        return [1,0]
    #                             [no cat, very doggo]
    elif word_label == 'dog': 
        print('dog')
        return [0,1]
    
#returns an numpy array of train and test data
def create_train_data():
    training_data = []
    for img in tqdm(SHORT_LIST_TRAIN):
        label = label_img(img)
        path = os.path.join(TRAIN_DIR,img)
        img = cv2.imread(path,cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        training_data.append([np.array(img),np.array(label)])
    shuffle(training_data)
    return training_data

def process_test_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR,img)
        img_num = img.split('.')[0]
        img = cv2.imread(path,cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        testing_data.append([np.array(img), img_num])
    shuffle(testing_data)
    return testing_data

labels = []
for i in SHORT_LIST_TRAIN:
    if 'dog' in i:
        labels.append(1)
    else:
        labels.append(0)

sns.countplot(labels)
plt.title('Cats and Dogs')

# Creating a Training Set Data
train = create_train_data()

# From Train Dividing X and Y
X = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,3)
Y = np.array([i[1] for i in train])

# Specify Model
from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D

NUM_CLASSES = 2
RESNET_WEIGHTS_PATH = 'input/dogs-vs-cats-redux-kernels-edition/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5' #importing a pretrained model
my_new_model = Sequential()
my_new_model.add(ResNet50(include_top=False, pooling='max', weights=RESNET_WEIGHTS_PATH))
my_new_model.add(Dense(NUM_CLASSES, activation='softmax'))

# Say not to train first layer (ResNet) model. It is already trained
my_new_model.layers[0].trainable = False

# Compile Model
my_new_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

# Model Sum
my_new_model.summary()

# Fit Model
history = my_new_model.fit(X, Y, validation_split=0.20, epochs=2, batch_size=64)

# Plot Results
X = n
p.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,3)
Y = np.array([i[1] for i in train])


plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# Testing Model on the Test Data Hand Sign

import matplotlib.pyplot as plt

# if you need to create the data:
test_data = process_test_data()
# if you already have some saved:
#test_data = np.load('test_data.npy')

fig=plt.figure()

for num,data in enumerate(test_data[:12]):
    # cat: [1,0]
    # dog: [0,1]
    
    img_num = data[1]
    img_data = data[0]
    
    y = fig.add_subplot(3,4,num+1)
    orig = img_data
    data = img_data.reshape(-1,IMG_SIZE,IMG_SIZE,3)
    #model_out = model.predict([data])[0]
    model_out = my_new_model.predict([data])[0]
    
    if np.argmax(model_out) == 1: str_label='Dog'
    else: str_label='Cat'
        
    y.imshow(orig)
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
    
plt.show()
prob = []
img_list = []
for data in tqdm(test_data):
        img_num = data[1]
        img_data = data[0]
        orig = img_data
        data = img_data.reshape(-1,IMG_SIZE,IMG_SIZE,3)
        model_out = my_new_model.predict([data])[0]
        img_list.append(img_num)
        prob.append(model_out[1])

# Submission of the model to csv
        
submission = pd.DataFrame({'id':img_list , 'label':prob})
print(submission.head())

submission.to_csv('./Transfer Learning/submission.csv', index=False)
sub_csv = pd.read_csv('./Transfer Learning/submission.csv')

# Saving model weight and model to .h5
from keras.models import load_model

model = my_new_model

model.save('model_cat_dog.h5')
model.save_weights('model_cat_dog_weights.h5')
