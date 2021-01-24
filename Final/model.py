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


# Split keras DataSet objects into image and tag sets:
x_train = np.concatenate([x for x, y in train_ds], axis=0)
y_train = np.concatenate([y for x, y in train_ds], axis=0)

x_test = np.concatenate([x for x, y in val_ds], axis=0)
y_test = np.concatenate([y for x, y in val_ds], axis=0)


# One hot encode the y vals to be able to make the confusion matrix:
from keras.utils import to_categorical

y_train_one_hot = to_categorical(y_train, num_classes= 39)
y_test_one_hot = to_categorical(y_test, num_classes= 39)

# Easier naming convention:
test_data = x_test
test_labels = y_test_one_hot

# Create predictions:
preds = model.predict(test_data)
np.argmax(preds, axis=1)

# Produce and display classification metrics (f1, recall, precision, accuracy):
classification_metrics = metrics.classification_report(test_labels, preds, target_names = class_names)
print(classification_metrics)

categorical_test_labels = pd.DataFrame(y_test_one_hot).idxmax(axis=1)
categorical_preds = pd.DataFrame(preds).idxmax(axis=1)


# Produce confusion matrix and display it:
confusion_matrix= confusion_matrix(categorical_test_labels, categorical_preds)


confusion_matrix = metrics.confusion_matrix(y_true=y_test_one_hot, y_pred=preds, labels=class_names)
print(confusion_matrix)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(confusion_matrix)
plt.title('Confusion matrix of the classifier')
fig.colorbar(cax)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()