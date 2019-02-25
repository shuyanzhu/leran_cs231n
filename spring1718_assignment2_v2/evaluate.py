import tensorflow as tf
import learn_keras

model = tf.keras.models.load_model('mode-04')
_, _, _, _, X_test, y_test = learn_keras.load_cifar10()
model.compile(optimizer=tf.train.AdamOptimizer(5e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
history = model.evaluate(X_test, y_test, batch_size=64)
predic = model.predict(X_test, batch_size=32)
print(predic.shape)
