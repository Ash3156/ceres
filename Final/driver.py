import numpy as np
import tensorflow as tf
from tensorflow import keras

# class_names = [
#     'Apple_scab',
#     'Apple_black_rot',
#     'Apple_cedar_apple_rust',
#     'Apple_healthy',
#     'Background_without_leaves',
#     'Blueberry_healthy',
#     'Cherry_powdery_mildew',
#     'Cherry_healthy',
#     'Corn_gray_leaf_spot',
#     'Corn_common_rust',
#     'Corn_northern_leaf_blight',
#     'Corn_healthy',
#     'Grape_black_rot',
#     'Grape_black_measles',
#     'Grape_leaf_blight',
#     'Grape_healthy',
#     'Orange_haunglongbing',
#     'Peach_bacterial_spot',
#     'Peach_healthy',
#     'Pepper_bacterial_spot',
#     'Pepper_healthy',
#     'Potato_early_blight',
#     'Potato_healthy',
#     'Potato_late_blight',
#     'Raspberry_healthy',
#     'Soybean_healthy',
#     'Squash_powdery_mildew',
#     'Strawberry_healthy',
#     'Strawberry_leaf_scorch',
#     'Tomato_bacterial_spot',
#     'Tomato_early_blight',
#     'Tomato_healthy',
#     'Tomato_late_blight',
#     'Tomato_leaf_mold',
#     'Tomato_septoria_leaf_spot',
#     'Tomato_spider_mites_two-spotted_spider_mite',
#     'Tomato_target_spot',
#     'Tomato_mosaic_virus',
#     'Tomato_yellow_leaf_curl_virus'
# ]


model = tf.keras.models.load_model('15epoch.h5')

def predict_new (path):
    img = keras.preprocessing.image.load_img(path, target_size=(250, 250))
    img_array = tf.expand_dims(keras.preprocessing.image.img_to_array(img), 0)
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    # print("Classified as {}. Model is {} percent confident in this classification.".format(class_names[np.argmax(score)], 100 * np.max(score)))
    return np.argmax(score)

# Iterate through user images and run them through predict_new:
