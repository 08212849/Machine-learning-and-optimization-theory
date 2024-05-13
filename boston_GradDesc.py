from sklearn.preprocessing import StandardScaler
import numpy as np
import random
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

# 从 OpenML 获取波士顿房价数据集
boston = fetch_openml(name='boston', version=1)  # version 可能需要根据实际情况调整
X, y = boston.data, boston.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # 特征缩放
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 生成随机颜色
def randomColor():
    hex_color = "#{:02X}{:02X}{:02X}".format(
        random.randint(0, 255),
        random.randint(0, 255),
        random.randint(0, 255)
    )
    return hex_color

class Network(object):
    def __init__(self, num_of_weights, optimizer, iterations=2000, learning_rate=0.25):
        self.w = np.random.randn(num_of_weights, 1)
        self.b = 0.
        self.optimizer = optimizer
        self.iterations = iterations
        # 初始化 momentum 所需的附加变量
        self.momentum_w = np.random.randn(num_of_weights, 1)  # 匹配权重的维度
        self.momentum_b = 0.
        # 初始化 Adagrad 所需的附加变量
        self.accum_grad_w = np.random.randn(num_of_weights, 1)
        self.accum_grad_b = 0.
        self.learning_rate = learning_rate
        # 初始化 Adam 所需的附加变量
        self.m_w = np.zeros((num_of_weights, 1))
        self.m_b = 0.
        self.v_w = np.zeros((num_of_weights, 1))
        self.v_b = 0.

        # losses = self.train(X_train_scaled, y_train, eta=0.1)
        # print(optimizer, 'loss:', losses)

    def forward(self, x):
        z = np.dot(x, self.w) + self.b
        z = z.squeeze()
        return z

    def loss(self, z, y):
        error = z - y
        num_samples = error.shape[0]
        cost = error * error
        cost = np.sum(cost) / num_samples
        return cost

    def gradient(self, x, y):
        z = self.forward(x)
        t = np.array(z - y)
        t = t[:, np.newaxis]
        gradient_w = t * x
        gradient_w = np.mean(gradient_w, axis=0)
        gradient_w = gradient_w[:, np.newaxis]
        gradient_b = (z - y)
        gradient_b = np.mean(gradient_b)
        return gradient_w, gradient_b

    def update_momentum(self, gradient_w, gradient_b, eta=0.01, beta=0.9):
        momentum_w = beta * self.momentum_w - eta * gradient_w
        momentum_b = beta * self.momentum_b - eta * gradient_b
        self.momentum_w = momentum_w
        self.momentum_b = momentum_b
        self.w = self.w + momentum_w
        self.b = self.b + momentum_b

    def update_sgd(self, gradient_w, gradient_b, eta=0.01):
        self.w = self.w - eta * gradient_w
        self.b = self.b - eta * gradient_b

    def update_adagrad(self, gradient_w, gradient_b, eps=1e-8):
        self.accum_grad_w += gradient_w ** 2
        self.accum_grad_b += gradient_b ** 2
        self.w -= self.learning_rate / (np.sqrt(self.accum_grad_w) + eps) * gradient_w
        self.b -= self.learning_rate / (np.sqrt(self.accum_grad_b) + eps) * gradient_b

    def update_adam(self, gradient_w, gradient_b, t, beta1=0.9, beta2=0.999, eps=1e-8):
        self.m_w = beta1 * self.m_w + (1 - beta1) * gradient_w
        self.m_b = beta1 * self.m_b + (1 - beta1) * gradient_b
        self.v_w = beta2 * self.v_w + (1 - beta2) * (gradient_w ** 2)
        self.v_b = beta2 * self.v_b + (1 - beta2) * (gradient_b ** 2)
        t += 1
        alpha_w = self.learning_rate * (np.sqrt(1 - beta1 ** t) ** 0.5 * (self.m_w / (np.sqrt(self.v_w) + eps)))
        alpha_b = self.learning_rate * (np.sqrt(1 - beta1 ** t) ** 0.5 * (self.m_b / (np.sqrt(self.v_b) + eps)))
        self.w -= alpha_w
        self.b -= alpha_b
        return t

    def train(self, x, y, eta=0.01):
        t = 0
        losses = []
        for i in range(self.iterations):
            z = self.forward(x)
            gradient_w, gradient_b = self.gradient(x, y)
            if self.optimizer == 'gd':
                self.update_sgd(gradient_w, gradient_b, eta)

            elif self.optimizer == 'momentum':
                beta = 0.9
                self.update_momentum(gradient_w, gradient_b, eta, beta)

            elif self.optimizer == 'adagrad':
                self.update_adagrad(gradient_w, gradient_b)

            elif self.optimizer == 'adam':
                t = self.update_adam(gradient_w, gradient_b, t)
            L = self.loss(z, y)
            losses.append(L)

        # self.drawLoss(losses)
        self.drawGDResult()
        print(self.optimizer, self.loss(z, y))
        return self.loss(z, y)

    def drawGDResult(self):
        colors = {'adam': 'cyan', 'adagrad': 'k', 'momentum': 'green', 'gd': 'b'}
        y_pred = np.dot(X_test_scaled, self.w) + self.b
        plt.plot(y_pred, c=colors[self.optimizer], label="Pred_" + self.optimizer)

    def drawLoss(self, losses):
        colors = {'adam': 'cyan', 'adagrad': 'k', 'momentum': 'r', 'gd': 'b'}
        plt.plot(losses, c=colors[self.optimizer], label="Loss_" + self.optimizer)


net_sgd = Network(13, 'gd')
net_sgd.train(X_train_scaled, y_train, eta=0.1)
print(net_sgd.w)

net_memont = Network(13, 'momentum')
net_memont.train(X_train_scaled, y_train, eta=0.1)

net_adam = Network(13, 'adam')
net_adam.train(X_train_scaled, y_train, eta=0.1)

net_adagrad = Network(13, 'adagrad', learning_rate=0.3)
net_adagrad.train(X_train_scaled, y_train, eta=0.1)


y_test = y_test.values
y_test = y_test[np.newaxis, :].transpose()
plt.plot(y_test, c="r", label="Actual")

# plt.locator_params(axis='x', nbins=15)  # x 轴自动调整，最多 nbins 个刻度
# plt.locator_params(axis='y', nbins=15)  # y 轴自动调整，最多 nbins 个刻度
plt.legend()
plt.show()