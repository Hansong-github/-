"""
    逻辑回归实现鸢尾花分类预测 （二分类）
"""
import pandas as pd
import sklearn.datasets as sd  # 加载鸢尾花数据集
import sklearn.linear_model as lm
import sklearn.model_selection as sm
import numpy as np
import math
import matplotlib.pyplot as plt

# 加载数据集
iris = sd.load_iris()
# print(iris.data.shape)
# print(iris.feature_names)  # 特征数据名
# print(iris.target.shape) #特征类别标签
# print(iris.target_names)#类别名称

# 方便查看，将输入集与类别标签整合在一起
data = pd.DataFrame(iris.data,
                    columns=iris.feature_names,
                    )
data['target'] = iris.target  # 添加一列标签数据
# print(data)

# 萼片可视化
data.plot.scatter(
    x='sepal length (cm)',  # 萼片长度
    y='sepal width (cm)',  # 萼片宽度
    c='target',  # 颜色映射
    cmap='brg'  # 映射颜色
)

# 花瓣可视化
data.plot.scatter(
    x='petal length (cm)',
    y='petal width (cm)',
    c='target',
    cmap='brg'
)
# plt.show()

# 数据集的划分
# 划分1类别和2类别
sub_data = data[data['target'] > 0]
print(sub_data)

# 整理输入集与输出集
x = sub_data.iloc[:, :-1]
y = sub_data.iloc[:, -1]

# 划分训练集与测试集
train_x, test_x, train_y, test_y = sm.train_test_split(
    x,  # 输入集
    y,  # 输出集
    test_size=0.1,  # 测试集占比
    random_state=7  # 随机种子
)

# 逻辑回归模型的构建
model = lm.LogisticRegression()

# 模型训练
model.fit(train_x, train_y)

# 模型预测
pred_test_y = model.predict(test_x)

print(pred_test_y)
print(test_y)

# 计算准确率
print('准确率：', (pred_test_y == test_y.values).sum() / test_y.size)

print('*' * 60)

"""
    基于逻辑回归实现鸢尾花数据集划分（多分类）
"""

# 多分类划分训练集与测试集
train_x, test_x, train_y, test_y = sm.train_test_split(
    x,  # 输入集
    y,  # 输出集
    test_size=0.2,  # 测试集占比
    random_state=7,
    stratify=y  # 按照y中的比例划分
)

# 构建模型
model = lm.LogisticRegression(solver="liblinear")

# 模型训练
model.fit(train_x,train_y)

# 模型预测
pred_test_y = model.predict(test_x)

# 评估
print("准确率：",(pred_test_y == test_y).sum() / test_y.size)
print(pred_test_y)
print(test_y.values)

















