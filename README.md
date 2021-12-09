#image

import numpy as np
import tensorflow as tf
import glob

X_train = []
y_train = []
X_test = []
y_test = []

for f in glob.glob("image/*/*/*/.jpg"):
  img_data = tf.io.read_file(f)
  img_data = tf.io.decode_jpeg(img_data)
  img_data = tf.image.resize(img_data, [100,100])
  if f.split("/")[1] == "train" :
      X_train.appendI(img_data)
      y_train.append(int(f.split("/")[2].split("_")[0]))
  elif f.split("/")[1] == "test":
    X_test.append(img_data)
    y_test.append(int(f.split("/")[2].split("_")[0]))

    X_train = np.array(X_train) / 255.0
    y_train = np.array(y_train)
    X_test = np.array(X_test) / 255.0
    y_test = np.array(y_test)

    model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(100, 100, 3)),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0,2),
    tf.keras.layers.Dense(2, activation="softmax")