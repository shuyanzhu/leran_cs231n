import numpy as np
import tensorflow as tf
# 加载数据
def load_cifar10():
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
    return X_train, y_train, X_val, y_val, X_test, y_test
# 建立模型
def inference(inputs):
    x = tf.layers.BatchNormalization()(inputs)
    x = tf.layers.Conv2D(16, [5, 5], activation='relu', padding='same')(x)
    x = tf.layers.BatchNormalization()(x)
    x = tf.layers.Conv2D(32, [3, 3], activation='relu', padding='same')(x)
    x = tf.layers.Flatten()(x)
    x = tf.layers.Dropout()(x)
    x = tf.layers.BatchNormalization()(x)
    x = tf.layers.Dense(120, activation='relu')(x)
    x = tf.layers.Dropout()(x)
    x = tf.layers.BatchNormalization()(x)
    outputs = tf.layers.Dense(10, activation='softmax')(x)
    return outputs
if __name__ == '__main__':
    # 导入数据
    X_train, y_train, X_val, y_val, X_test, y_test = load_cifar10()
    train_dset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    validation_dset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    # 构建模型
    inputs = tf.keras.Input(shape=(32, 32, 3))
    model = tf.keras.Model(inputs=inputs, outputs=inference(inputs))
    model.compile(optimizer=tf.train.AdamOptimizer(5e-4),
                loss = 'sparse_categorical_crossentropy',
                metrics=['accuracy'])
    model.fit(train_dset, batch_size=64, epochs=5, validation_data=validation_dset)
    
    
