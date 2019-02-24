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
    """
    Simple training loop for use with models defined using tf.keras. It trains
    a model for one epoch on the CIFAR-10 training set and periodically checks
    accuracy on the CIFAR-10 validation set.

    Inputs:
    - model_init_fn: A function that takes no parameters; when called it
      constructs the model we want to train: model = model_init_fn()
    - optimizer_init_fn: A function which takes no parameters; when called it
      constructs the Optimizer object we will use to optimize the model:
      optimizer = optimizer_init_fn()
    - num_epochs: The number of epochs to train for

    Returns: Nothing, but prints progress during trainingn
    """
    tf.reset_default_graph()
    with tf.device(device):
        # Construct the computational graph we will use to train the model. We
        # use the model_init_fn to construct the model, declare placeholders for
        # the data and labels
        x = tf.placeholder(tf.float32, [None, 32, 32, 3])
        y = tf.placeholder(tf.int32, [None])

        # We need a place holder to explicitly specify if the model is in the training
        # phase or not. This is because a number of layers behaves differently in
        # training and in testing, e.g., dropout and batch normalization.
        # We pass this variable to the computation graph through feed_dict as shown below.
        is_training = tf.placeholder(tf.bool, name='is_training')

        # Use the model function to build the forward pass.
        scores = model_init_fn(x)

        # Compute the loss like we did in Part II
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=scores)
        loss = tf.reduce_mean(loss)

        # Use the optimizer_fn to construct an Optimizer, then use the optimizer
        # to set up the training step. Asking TensorFlow to evaluate the
        # train_op returned by optimizer.minimize(loss) will cause us to make a
        # single update step using the current minibatch of data.

        # Note that we use tf.control_dependencies to force the model to run
        # the tf.GraphKeys.UPDATE_OPS at each training step. tf.GraphKeys.UPDATE_OPS
        # holds the operators that update the states of the network.
        # For example, the tf.layers.batch_normalization function adds the running mean
        # and variance update operators to tf.GraphKeys.UPDATE_OPS.
        train_op = tf.train.AdamOptimizer(5e-4).minimize(loss)

    # Now we can run the computational graph many times to train the model.
    # When we call sess.run we ask it to evaluate train_op, which causes the
    # model to update.
    with tf.Session() as sess:
        xx, yy, initializer, xs, ys = gen_batch(X_train, y_train)
        sess.run(initializer, feed_dict={xx: X_train, yy: y_train})
        sess.run(tf.global_variables_initializer())
        t = 0
        for epoch in range(num_epochs):
            print('Starting epoch %d' % epoch)
            while True:
                try:
                    x_np, y_np = sess.run([xs, ys])
                    feed_dict = {x: x_np, y: y_np, is_training: 1}
                    loss_np, _ = sess.run([loss, train_op], feed_dict=feed_dict)
                    if t % print_every == 0:
                        print('Iteration %d, loss = %.4f' % (t, loss_np))
                        check_accuracy(sess, val_dset, x, scores, is_training=is_training)
                        print()
                    t += 1
                except tf.errors.OutOfRangeError:
                    break
if __name__ == '__main__':
    # 导入数据
    X_train, y_train, X_val, y_val, X_test, y_test = load_cifar10()
    train_dset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    validation_dset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    # 构建模型
    inputs = tf.keras.Inputs(shape=(32, 32, 3))
    model = tf.keras.Model(inputs=inputs, inference(inputs))
    model.compile(optimizer=tf.train.AdamOptimizer(5e-4),
                loss = 'sparse_categorical_crossentropy',
                metrics=['accuracy'])
    model.fit(train_dset, batch_size=64, epochs=5)
    
    
