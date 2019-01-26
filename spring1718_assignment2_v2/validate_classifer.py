# coding=utf-8
# 加载数据
import numpy as np
import matplotlib.pyplot as plt
from cs231n.classifiers.cnn import *
from cs231n.data_utils import get_CIFAR10_data
from cs231n.gradient_check import eval_numerical_gradient_array, eval_numerical_gradient
from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.solver import Solver

cifar10_path = 'cs231n/datasets/cifar-10-batches-py'
data = get_CIFAR10_data()

for k, v in data.items():
    print('%s:' % k, v.shape)

from cs231n.classifiers import cnn
num_train = 500
small_data = {
    'X_train': data['X_train'][:num_train],
    'y_train': data['y_train'][:num_train],
    'X_val': data['X_val'],
    'y_val': data['y_val'],
    'X_test': data['X_test'],
    'y_test': data['y_test']
}
classifer = ThreeLayerConvNet(hidden_dim=200, weight_scale=1e-2, reg=0.001)
solver = Solver(classifer, small_data,
                num_epochs=20, batch_size=100,
                update_rule='adam', optim_config={'learning_rate': 1e-4},
                verbose=True, print_every=10000
                )
# num_training = 50000   # 训练集大小
# mask = np.random.choice(50000, 5000, replace=False)
# X_train = X_train[mask]
# y_train = y_train[mask]
# num_test = 500 # 测试集大小
# mask = np.random.choice(10000, 100, replace=False)
# X_test = X_test[mask]
# y_test = y_test[mask]

# 训练模型
# X_train = np.reshape(X_train, (X_train.shape[0], -1)) # 1维展开
# X_train = np.hstack([X_train, np.ones([X_train.shape[0], 1])])
# X_test = np.reshape(X_test, (X_test.shape[0], -1)) # 1维展开
# X_test = np.hstack([X_test, np.ones([X_test.shape[0], 1])])
# num_class = 10
# W = np.random.randn(X_train.shape[1], num_class) * 0.001
# # 检查数值梯度和解析梯度
# from cs231n.classifiers import softmax_loss_naive, softmax_loss_vectorized
# loss, grad = softmax_loss_naive(W, X_train, y_train, 0.5)
# from cs231n.gradient_check import grad_check_sparse
# f = lambda w:softmax_loss_vectorized(w, X_train, y_train, 0.5)[0]
# grad_check_sparse(f, W, grad)
# from cs231n.classifiers import Softmax
# classifer = Softmax()
# loss_hist = classifer.train(X_train, y_train, verbose=True,num_iters=5000, batch_size=100)
# plt.plot(loss_hist)
# plt.xlabel('Step')
# plt.ylabel('Loss')
# plt.show()
# 泛化准确率
# from
# y_pred = classifer.predict(X_test)
# accuracy = np.mean(y_pred == y_test)
# print("Test accuracy:", accuracy)

# from cs231n.classifiers import KNearestNeighbor, KNN_test, KNN_train
# classifer = KNearestNeighbor()
# KNN_train(classifer, X_train, y_train) # 训练
# KNN_test(classifer, X_test, y_test) # 泛化准确率
