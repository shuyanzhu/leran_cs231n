import numpy as np

# 训练和调参
def KNN_train(classifer, X_train, y_train):
  # 将图片展开为一维数组, 训练模型, 5折交叉检验
  X_train = np.reshape(X_train, (X_train.shape[0], -1))
  k_choices = [1, 3, 5, 8, 10, 12, 15]
  num_k_fold = 5
  val_accuracy = []
  X_train_folds = np.split(X_train, num_k_fold)
  y_train_folds = np.split(y_train, num_k_fold)
  for k in k_choices:
    k_val_accuracy = []
    for i in range(num_k_fold):
      tempX = X_train_folds[:]
      tempy = y_train_folds[:]
      X_val = tempX.pop(i)
      y_val = tempy.pop(i)
      # 训练模型
      tempX = np.array([y for x in tempX for y in x])
      tempy = np.array([y for x in tempy for y in x])
      classifer.train(tempX, tempy)
      # 模型正确率
      y_val_pred = classifer.predict(X_val, k)
      k_val_accuracy.append(np.flatnonzero(y_val_pred == y_val).shape[0] / y_val.shape[0])
    val_accuracy.append(k_val_accuracy)
  best = 0
  for i, k in enumerate(k_choices):
    accuracy = sum(val_accuracy[i]) / num_k_fold
    if accuracy > best:
      best = accuracy
      classifer.K = k
    print('Validate accuracy %.3f of k = %d' % (accuracy, k))
  print('Best K: %d have an accuracy %.3f' % (classifer.K, best))
  classifer.train(X_train, y_train)

# 得到测试正确率(进一步用于假设检验是否为可以采用的泛化正确率)
def KNN_test(classifer, X_test, y_test):
  X_test = np.reshape(X_test, (X_test.shape[0], -1))
  y_test_pred = classifer.predict(X_test, classifer.K)  # 10 nearest neighbors
  print('Test accuracy:', np.flatnonzero(y_test_pred == y_test).shape[0] / y_test.shape[0])

class KNearestNeighbor(object):
  """ a kNN classifier with L2 distance """

  def __init__(self):
    pass

  def train(self, X, y):
    """
    Train the classifier. For k-nearest neighbors this is just 
    memorizing the training data.

    Inputs:
    - X: A numpy array of shape (num_train, D) containing the training data
      consisting of num_train samples each of dimension D.
    - y: A numpy array of shape (N,) containing the training labels, where
         y[i] is the label for X[i].
    """
    self.X_train = X
    self.y_train = y
    
  def predict(self, X, k=1, num_loops=0):
    """
    Predict labels for test data using this classifier.

    Inputs:
    - X: A numpy array of shape (num_test, D) containing test data consisting
         of num_test samples each of dimension D.
    - k: The number of nearest neighbors that vote for the predicted labels.
    - num_loops: Determines which implementation to use to compute distances
      between training points and testing points.

    Returns:
    - y: A numpy array of shape (num_test,) containing predicted labels for the
      test data, where y[i] is the predicted label for the test point X[i].  
    """
    if num_loops == 0:
      dists = self.compute_distances_no_loops(X)
    elif num_loops == 1:
      dists = self.compute_distances_one_loop(X)
    elif num_loops == 2:
      dists = self.compute_distances_two_loops(X)
    else:
      raise ValueError('Invalid value %d for num_loops' % num_loops)

    return self.predict_labels(dists, k=k)

  def compute_distances_two_loops(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using a nested loop over both the training data and the 
    test data.

    Inputs:
    - X: A numpy array of shape (num_test, D) containing test data.

    Returns:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      is the Euclidean distance between the ith test point and the jth training
      point.
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in range(num_test):
      for j in range(num_train):
        #####################################################################
        # TODO:                                                             #
        # Compute the l2 distance between the ith test point and the jth    #
        # training point, and store the result in dists[i, j]. You should   #
        # not use a loop over dimension.                                    #
        #####################################################################
        dists[i, j] = np.linalg.norm(X[i]-self.X_train[j])
        # dists[i, j] = np.sqrt(np.sum(np.square(X[i] - self.X_train[j])))
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
    return dists

  def compute_distances_one_loop(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using a single loop over the test data.

    Input / Output: Same as compute_distances_two_loops
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in range(num_test):
      #######################################################################
      # TODO:                                                               #
      # Compute the l2 distance between the ith test point and all training #
      # points, and store the result in dists[i, :].                        #
      #######################################################################
      dists[i] = np.linalg.norm(X[i]-self.X_train, axis=1)
      #######################################################################
      #                         END OF YOUR CODE                            #
      #######################################################################
    return dists

  def compute_distances_no_loops(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using no explicit loops.

    Input / Output: Same as compute_distances_two_loops
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train)) 
    #########################################################################
    # TODO:                                                                 #
    # Compute the l2 distance between all test points and all training      #
    # points without using any explicit loops, and store the result in      #
    # dists.                                                                #
    #                                                                       #
    # You should implement this function using only basic array operations; #
    # in particular you should not use functions from scipy.                #
    #                                                                       #
    # HINT: Try to formulate the l2 distance using matrix multiplication    #
    #       and two broadcast sums.                                         #
    #########################################################################
    dists = -2 * np.dot(X, self.X_train.T)
    sumX = np.sum(np.square(X), axis=1, keepdims=True)
    sumXt = np.sum(np.square(self.X_train), axis=1)
    dists = np.add(dists, sumX)
    dists = np.add(dists, sumXt)
    dists = np.sqrt(dists)
    #########################################################################
    #                         END OF YOUR CODE                              #
    #########################################################################
    return dists

  def predict_labels(self, dists, k=1):
    """
    Given a matrix of distances between test points and training points,
    predict a label for each test point.

    Inputs:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      gives the distance betwen the ith test point and the jth training point.

    Returns:
    - y: A numpy array of shape (num_test,) containing predicted labels for the
      test data, where y[i] is the predicted label for the test point X[i].  
    """
    num_test = dists.shape[0]
    y_pred = np.zeros(num_test)
    for i in range(num_test):
      # A list of length k storing the labels of the k nearest neighbors to
      # the ith test point.
      closest_y = []
      #########################################################################
      # TODO:                                                                 #
      # Use the distance matrix to find the k nearest neighbors of the ith    #
      # testing point, and use self.y_train to find the labels of these       #
      # neighbors. Store these labels in closest_y.                           #
      # Hint: Look up the function numpy.argsort.                             #
      #########################################################################
      k_nearest = np.argsort(dists[i])[:k]
      closest_y = self.y_train[k_nearest]
      #########################################################################
      # TODO:                                                                 #
      # Now that you have found the labels of the k nearest neighbors, you    #
      # need to find the most common label in the list closest_y of labels.   #
      # Store this label in y_pred[i]. Break ties by choosing the smaller     #
      # label.                                                                #
      #########################################################################
      y_pred[i] = np.argmax(np.bincount(closest_y))
      #########################################################################
      #                           END OF YOUR CODE                            # 
      #########################################################################

    return y_pred

