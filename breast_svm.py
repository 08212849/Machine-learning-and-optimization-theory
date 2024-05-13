import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

data = pd.read_csv("./数据集/data.csv")

# 数据预处理
pd.set_option('display.max_columns', None)

# 将特征字段分成3组
features_mean= list(data.columns[2:12])
features_se= list(data.columns[12:22])
features_worst=list(data.columns[22:32])

# 数据清洗
# 删除ID列
data.drop("id",axis=1,inplace=True)
# 将B良性替换为0，M恶性替换为1
data['diagnosis']=data['diagnosis'].map({'M':1,'B':0})

# 特征选择
features_remain = ['radius_mean','texture_mean', 'smoothness_mean','compactness_mean','symmetry_mean', 'fractal_dimension_mean'] 

# 抽取30%的数据作为测试集，其余作为训练集
train, test = train_test_split(data, test_size=0.3)
# 抽取特征选择的数值作为训练和测试数据
train_X = train[features_remain]
train_y = train['diagnosis']
test_X = test[features_remain]
test_y = test['diagnosis']


# 用热力图呈现features_mean字段之间的相关性
# corr = data[features_mean].corr()
# plt.figure(figsize=(14,14))
# annot=True显示每个方格的数据
# sns.heatmap(corr, annot=True)
# plt.show()

# 采用Z-Score规范化数据，保证每个特征维度的数据均值为0，方差为1
ss = StandardScaler()
train_X = ss.fit_transform(train_X)
test_X = ss.transform(test_X)

kernels = ['linear', 'poly', 'rbf', 'sigmoid']
# 存储结果的字典
results = {}

for kernel in kernels:
    # 创建SVM分类器实例
    svm = SVC(kernel=kernel)
    svm.fit(train_X, train_y)

    y_pred = svm.predict(test_X)
    accuracy = accuracy_score(test_y, y_pred)
    results[kernel] = accuracy

# 打印每种核函数的准确率
for kernel, accuracy in results.items():
    print(f"Accuracy with {kernel} kernel: {accuracy:.2f}")