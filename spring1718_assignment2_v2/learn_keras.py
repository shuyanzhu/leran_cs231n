import numpy as np
import tensorflow as tf
# 加载数据
cifar10 = tf.keras.datasets.cifar10.load_data()
(X_train, y_train), (X_test, y_test) = cifar10
X_train = X_train.astype(np.float32)
y_train = y_train.astype(np.int32).flatten()
X_test = X_test.astype(np.float32)
y_test = y_test.astype(np.int32).flatten()
val_size = 1000
X_val = X_train[:val_size]
y_val = y_train[:val_size]
X_train = X_train[val_size:]
y_train = y_train[val_size:]
mean_pixel = X_train.mean(axis=(0, 1, 2), keepdims=True)
std_pixel = X_train.std(axis=(0, 1, 2), keepdims=True)
X_train = (X_train - mean_pixel) / std_pixel
X_val = (X_val - mean_pixel) / std_pixel
X_test = (X_test - mean_pixel) / std_pixel
print('Trian data size', X_train.shape)
print('Train labels size', y_train.shape)
print('Validation data size', X_val.shape)
print('Validation labels size', y_val.shape)
print('Test data size', X_test.shape)
print('Test labels size', y_test.shape)
# 建立模型
inputs = tf.keras.Input(shape=(32, 32, 3))
x = tf.layers.BatchNormalization()(inputs)
x = tf.layers.Conv2D(32, [5, 5], activation='relu', padding='same')(x)
x = tf.layers.MaxPooling2D([2, 2], [1, 1])(x)
x = tf.layers.BatchNormalization()(x)
x = tf.layers.Conv2D(64, [5, 5], activation='relu', padding='same')(x)
x = tf.layers.MaxPooling2D([2, 2], [1, 1])(x)
x = tf.layers.BatchNormalization()(x)
x = tf.layers.Conv2D(128, [3, 3], activation='relu', padding='same')(x)
x = tf.layers.MaxPooling2D([2, 2], [1, 1])(x)
x = tf.layers.Flatten()(x)
x = tf.layers.Dropout()(x)
x = tf.layers.BatchNormalization()(x)
x = tf.layers.Dense(120, activation='relu')(x)
x = tf.layers.Dropout()(x)
x = tf.layers.BatchNormalization()(x)
predictions = tf.layers.Dense(10, activation='softmax')(x)

model  = tf.keras.Model(inputs=inputs, outputs=predictions)
model.compile(optimizer=tf.train.AdamOptimizer(8e-5),
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=['accuracy'])
# 训练模型
model.fit(X_train, y_train, batch_size=64, epochs=10, validation_data=(X_val, y_val))
