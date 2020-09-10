'''
The model provided in the repo was trained on a combination of FER+ and AffectNet datasets with 75% accuracy on the test set.
    

To Train:
1. Download FER+ dataset through GitHub.
2. Extract the images of the original FER2013 dataset.
3. Place all Train, Test and Validation images into the single folder in "current_directory/dataset/images"
4. Run this model.py file.

FER+ has less misclassification error than FER2013 as there is no incorrect labelling.
The state-of-the-art techniques used in training FER2013 can be implemented here to increase accuracy
of model. Transfer Learning with other dataset-trained models and Fine-tuning those will lead to higher accuracy.

'''



import pandas as pd

#data
data = pd.read_csv("./dataset/fer2013new.csv")
data_prep = data.drop(['fear','contempt','unknown','NF','Usage','disgust'], axis = 1) 
data_prep['Emotion'] = data_prep.drop(['Image name'], axis = 1).idxmax(axis=1, skipna=True) #empty row is deleted
data_prep.rename(columns = {'Image name':'Image_name'}, inplace = True) 
data_prep['Emotion'] = data_prep['Emotion'].replace({'neutral':'0','happiness':'1','sadness':'2','surprise':'3','anger':'4'})
file_emotion = pd.Series(data_prep.Emotion.values,index=data_prep.Image_name).to_dict()

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D,BatchNormalization
import numpy as np
import os
from keras.preprocessing import image
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

x = []
y = []

### loading data
labels = {"Neutral":0,"Happy":1,"Sad":2,"Surprise":3,"Angry":4}

for key, value in file_emotion.items():	
	if key in os.listdir("./dataset/images"):
		img = image.load_img("./dataset/images/"+key, target_size=(48,48,1), color_mode = "grayscale")
		img = image.img_to_array(img)
		img = img/255.0
		x.append(img)
		y.append(value)
x = np.array(x)

# # Split the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, shuffle= True)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# CNN model:

model = Sequential()
model.add(Conv2D(64, (3, 3), input_shape=(48,48,1)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3)))
model.add(Dropout(0.3))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

model.add(Dense(128))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Dense(64))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(Dense(5)) #num_classes = 5
model.add(Activation('softmax'))

print(model.summary())

model.compile(loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])

model.fit(x_train, y_train, epochs=15, batch_size=64)
scores = model.evaluate(x_test, y_test)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
model.save('./expression.model')