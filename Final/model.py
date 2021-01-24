import pandas as pd
import numpy as np
import itertools
import keras
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.models import Sequential
from keras import optimizers
from keras.preprocessing import image
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
import datetime
import time
import tensorflow as tf
from tensorflow import keras
import os
import tempfile
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pandas as pd
import seaborn as sns
import glob
import cv2
from skimage.transform import resize
from sklearn.model_selection import train_test_split

# Process Data - [Split Train Test and Encode]:
data_dir = r"Plant_leave_diseases_dataset_with_augmentation"
batch_size = 100
img_height = 250
img_width = 250

# Define training data and testing data with 20/80 split and random seeding
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

# Verify class names:
class_names = train_ds.class_names
print(class_names)

# Create neural network:
model = tf.keras.models.Sequential([
  layers.BatchNormalization(),
  layers.Conv2D(32, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(128, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(256, activation='relu'),
  layers.Dense(len(class_names), activation= 'softmax')
])

# Compile Model:
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Fit with 15 epochs:
fitted = model.fit(training_data = train_ds,
                   validation_data= val_ds,
                   epochs = 15)

# Plot accuracy metrics from training:
plt.plot(fitted.history['loss'], label = 'loss')
plt.plot(fitted.history['accuracy'], label = 'accuracy')
plt.legend()
plt.show()

# Save model for future use:
# model.save('models/15epoch.h5')

# Print accuracy metrics:
loss, accuracy = model.evaluate(val_ds, verbose=2)
print('accuracy: {:5.2f}%'.format(100 * accuracy))
