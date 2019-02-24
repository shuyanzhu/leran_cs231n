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
X_train, y_train, X_val, y_val, X_test, y_test = load_cifar10()
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
    net = tf.layers.Dense(10)(x)
    return net
def gen_batch(features, labels):
    x = tf.placeholder(features.dtype, features.shape)
    y = tf.placeholder(labels.dtype, labels.shape)
    dset = tf.data.Dataset.from_tensor_slices((x, y))
    batch_dset = dset.batch(100)

    iterator = batch_dset.make_initializable_iterator()
    xs, ys = iterator.get_next()
    return x, y, iterator.initializer, xs, ys
class Dataset(object):
    def __init__(self, X, y, batch_size, shuffle=False):
        """
        Construct a Dataset object to iterate over data X and labels y

        Inputs:
        - X: Numpy array of data, of any shape
        - y: Numpy array of labels, of any shape but with y.shape[0] == X.shape[0]
        - batch_size: Integer giving number of elements per minibatch
        - shuffle: (optional) Boolean, whether to shuffle the data on each epoch
        """
        assert X.shape[0] == y.shape[0], 'Got different numbers of data and labels'
        self.X, self.y = X, y
        self.batch_size, self.shuffle = batch_size, shuffle

    def __iter__(self):
        N, B = self.X.shape[0], self.batch_size
        idxs = np.arange(N)
        if self.shuffle:
            np.random.shuffle(idxs)
        return iter((self.X[i:i + B], self.y[i:i + B]) for i in range(0, N, B))
val_dset = Dataset(X_val, y_val, batch_size=64, shuffle=False)
def check_accuracy(sess, dset, x, scores, is_training=None):
    """
    Check accuracy on a classification model.

    Inputs:
    - sess: A TensorFlow Session that will be used to run the graph
    - dset: A Dataset object on which to check accuracy
    - x: A TensorFlow placeholder Tensor where input images should be fed
    - scores: A TensorFlow Tensor representing the scores output from the
      model; this is the Tensor we will ask TensorFlow to evaluate.

    Returns: Nothing, but prints the accuracy of the model
    """
    num_correct, num_samples = 0, 0
    for x_batch, y_batch in dset:
        feed_dict = {x: x_batch, is_training: 0}
        scores_np = sess.run(scores, feed_dict=feed_dict)
        y_pred = scores_np.argmax(axis=1)
        num_samples += x_batch.shape[0]
        num_correct += (y_pred == y_batch).sum()
    acc = float(num_correct) / num_samples
    print('Got %d / %d correct (%.2f%%)' % (num_correct, num_samples, 100 * acc))
print_every = 50
device = '/cpu:0'
def train_part34(model_init_fn, num_epochs=1):
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
train_part34(inference, 20)
