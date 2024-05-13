# 实验一

## 1、数据集及预处理

波士顿房价数据集是著名的统计学和机器学习基准数据集，它包含了506个样本，每个样本都有13个特征变量，这些变量可能影响房屋价格。这些特征包括犯罪率、地方财产税率、学生-教师比例等，而目标变量是房屋价格的中位数。在本次实验中，采用线性回归模型来预测基于这些因素的房屋价格。

<img src="C:/Users/86199/AppData/Roaming/Typora/typora-user-images/image-20240505153816734.png" alt="image-20240505153816734" style="zoom: 67%;" />

将数据集划分为训练集和测试集两部分，比例为8:2。

为了消除不同量纲和量级的影响，提升模型的泛化能力和算法的收敛速度，对波士顿房价数据集中的特征进行了标准归一化处理。这一过程通过减去特征的均值并除以其标准差，将数据转换至一个均值为0、方差为1的正态分布。通过这种方式，确保各个特征在模型训练中具有相同的重要性权重，避免了某些特征由于数值范围大而对模型产生不成比例的影响。此外，对于梯度下降，缩放还可以加速收敛过程。

```python
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_openml

# 从 OpenML 获取波士顿房价数据集
boston = fetch_openml(name='boston', version=1) 
X, y = boston.data, boston.target
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

## 2、模型构建

- 以类的方式实现网络，使用时可以生成多个模型示例。

- 简单的类成员变量有w和b，在类初始化函数时初始化变量。

- `__init__` 方法初始化了模型的权重和偏置，同时设置了学习率。

- `forward` 方法实现了模型的前向计算，它接受输入数据 `X` 并返回预测值。

- `loss` 方法计算了当前权重和偏置下的损失函数值。

- `backward` 方法实现了反向传播算法，计算了损失函数关于权重和偏置的梯度。

- `update` 方法根据计算出的梯度更新权重和偏置，可根据不同优化方式设置函数。

- `train` 方法负责整个训练过程，包括前向计算、反向传播、参数更新以及损失函数的计算和打印。

  整个网络实现结构如图：

```python
class Network(object):
    def __init__(self, num_of_weights, optimizer, iterations=5000, learning_rate=0.25):...

    def forward(self, x):...

    def loss(self, z, y):...

    def gradient(self, x, y):...

    def update_momentum(self, gradient_w, gradient_b, eta=0.01, beta=0.9):...

    def update_gd(self, gradient_w, gradient_b, eta=0.01):...

    def update_adagrad(self, gradient_w, gradient_b, eps=1e-8):...
    
    def update_adam(self, gradient_w, gradient_b, t, beta1=0.09, beta2=0.05, eps=1e-8):...

    def train(self, x, y, eta=0.01):...

    def drawGDResult(self):...

    def drawLoss(self, losses):...
```

## 3、损失函数

在回归问题中，均方误差（Mean Squared Error, MSE）是一种常用的损失函数，用于衡量模型预测值与实际值之间的差异。它是预测误差的度量，可以为回归模型的训练提供指导。

均方误差是所有预测误差（残差）平方的平均值，其公式定义如下：

![image-20240505204135830](C:/Users/86199/AppData/Roaming/Typora/typora-user-images/image-20240505204135830.png)

其中：

- 𝑛*n* 是样本的总数。
- 𝑦𝑖*y**i* 是第 𝑖*i* 个样本的实际观测值。
- 𝑦^𝑖*y*^*i* 是第 𝑖*i* 个样本的预测值。

均方误差对较大的误差给予更大的惩罚，因为它对误差平方求平均。这意味着它对异常值（outliers）敏感，可能会因为个别大的误差而显著增加损失。同时也是一个可微的函数，这使得它适合于使用梯度下降等优化算法进行最小化。计算相对简单，且易于理解和实现。代码实现如下：

```python
    def loss(self, z, y):
        error = z - y
        num_samples = error.shape[0]
        cost = error * error
        cost = np.sum(cost) / num_samples
        return cost
```

## 4、优化算法选择

梯度下降（Gradient Descent, GD）、Adam（自适应矩估计）、Momentum（动量）和 AdaGrad（自适应梯度算法） 是常用的梯度下降优化算法，用于在机器学习模型训练过程中调整参数，下面对这些方法进行实验比较。

### 4.1 GD

梯度下降是一种一阶优化算法，用于最小化损失函数。它通过计算损失函数相对于模型参数的梯度，并沿着梯度的反方向更新参数来工作。实现简单，对于学习率的选择很关键，如果设置不当，可能导致超调或收敛到次优解；对于非凸函数，可能会陷入局部最小值。

更新函数如下：

```python
    def update_gd(self, gradient_w, gradient_b, eta=0.01):
        self.w = self.w - eta * gradient_w
        self.b = self.b - eta * gradient_b
```

### 4.2 Momentum

Momentum 是一种优化梯度下降的算法，通过将当前梯度与之前梯度的指数加权平均结合起来，减少梯度更新的抖动。对于具有高曲率的区域，可以更快地移动；对于平坦的区域，则减速。此外需要调整额外的参数（动量系数）。

更新规则如下：

![image-20240505211208197](C:/Users/86199/AppData/Roaming/Typora/typora-user-images/image-20240505211208197.png)

其中，𝑚𝑡 是动量项，𝛽是动量系数，实现如下：

```python
    def update_momentum(self, gradient_w, gradient_b, eta=0.01, beta=0.9):
        momentum_w = beta * self.momentum_w - eta * gradient_w
        momentum_b = beta * self.momentum_b - eta * gradient_b
        self.momentum_w = momentum_w
        self.momentum_b = momentum_b
        self.w = self.w + momentum_w
        self.b = self.b + momentum_b
```

### 4.3 Adam

Adam 是一种结合了动量和 RMSProp 思想的优化算法。Adam 结合了动量方法中的概念，即考虑过去梯度的指数衰减平均，使得更新方向更加平滑，同时结合 RMSProp 中的概念，即考虑过去梯度的平方的指数衰减平均，使得学习率自适应调整。它计算了梯度的一阶矩（动量）和二阶矩（方差），并使用这些矩来调整每个参数的学习率。

更新规则如下：

![image-20240505211730682](C:/Users/86199/AppData/Roaming/Typora/typora-user-images/image-20240505211730682.png)

其中，𝑚𝑡 和 𝑣𝑡 分别是梯度的一阶和二阶矩估计，𝛽1 和 𝛽2 是衰减率，实现如下：

```python
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
```

### 4.4 Adagrad

AdaGrad 通过为每个参数独立地调整学习率来优化梯度下降。它累积历史梯度的平方，并用这些信息来调整每个参数的学习率。缺点是累积梯度可能会导致学习率过早地变得非常小，从而减慢学习过程或导致难以收敛。

更新规则如下：

![image-20240505212122800](C:/Users/86199/AppData/Roaming/Typora/typora-user-images/image-20240505212122800.png)

其中，𝐺𝑡是累积的梯度平方和，𝜖是小常数，避免分母为零。实现如下：

```python
    def update_adagrad(self, gradient_w, gradient_b, eps=1e-8):
        self.accum_grad_w += gradient_w ** 2
        self.accum_grad_b += gradient_b ** 2
        self.w -= self.learning_rate / (np.sqrt(self.accum_grad_w) + eps) * gradient_w
        self.b -= self.learning_rate / (np.sqrt(self.accum_grad_b) + eps) * gradient_b
```

### 4.5 算法对比

在迭代次数分为50、500、5000的情况下，不同优化算法预测结果如图。

![Figure_50](D:/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E5%8F%8A%E4%BC%98%E5%8C%96/%E5%AE%9E%E9%AA%8C%E7%BB%93%E6%9E%9C/Figure_50.png)

![Figure_500](D:/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E5%8F%8A%E4%BC%98%E5%8C%96/%E5%AE%9E%E9%AA%8C%E7%BB%93%E6%9E%9C/Figure_500.png)

![Figure_2000](D:/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E5%8F%8A%E4%BC%98%E5%8C%96/%E5%AE%9E%E9%AA%8C%E7%BB%93%E6%9E%9C/Figure_2000.png)

momentum loss: 21.656081762580687
sgd loss: 21.9054372169007
adam loss: 22.83821015541945
adagrad loss: 21.64172587614321