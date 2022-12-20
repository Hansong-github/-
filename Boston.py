"""
    波士顿地区房屋价格预测
    根据13个特征值预测房屋价格（回归问题）
"""
# 决策树模型
# 加载数据，读取样本集
import pandas as pd
import sklearn.datasets as sd
import sklearn.model_selection as ms
import sklearn.ensemble as se  # 正向激励以及GBDT模块
import sklearn.metrics as sm  # 模型评估模块
import sklearn.tree as st  # 决策树模块
import sklearn.utils as su  # 数据处理模块

boston = sd.load_boston()
# print(boston.data.shape)
print(boston.feature_names)
# print(boston.target.shape)

# 数据预处理，样本随机化（消除样本顺序的影响）
x, y = su.shuffle(
    boston.data,  # 样本特征
    boston.target,  # 样本的标签
    random_state=7,  # 随机种子（产生随机数的初始值）
)

# 划分训练集（80%）与测试集（20%）
train_size = int(len(x) * 0.8)  # 训练集样本数量

train_x = x[:train_size]
test_x = x[train_size:]
train_y = y[:train_size]
test_y = y[train_size:]

# 定义模型
model = st.DecisionTreeRegressor(max_depth=5)  # 最大深度
model.fit(train_x, train_y)
pred_test_y = model.predict(test_x)  # 使用测试集预测

# 模型评估
print('r2得分：', sm.r2_score(test_y, pred_test_y))

# 特征重要性
fi = model.feature_importances_
print('fi:', fi)

# 特征重要性可视化
# plt.figure('Feature importances', facecolor='lightgray')
# plt.plot()
# plt.title('DT Feature', fontsize=16)
# plt.ylabel('Feature importances', fontsize=14)
# plt.grid(linestyle=":", axis='x')
# x = np.arange(fi.size)
# sorted_idx = fi.argsort()[::-1]  # 重要性排序（倒序）
# fi = fi[sorted_idx]  # 根据排序索引重新排特征值
# plt.xticks(x, boston.feature_names[sorted_idx])
# plt.bar(x, fi, 0.4, color='orange', label='DT Feature importances')

# plt.legend()
# plt.tight_layout()
# plt.show()
# print('*' * 60)
print('*' * 60)
# 利用adaboosting（正向激励树）模型实现波士顿房价预测

# 定义模型
model_base = st.DecisionTreeRegressor(max_depth=5)
model = se.AdaBoostRegressor(
    model_base,  # 子模块模型
    n_estimators=400,  # 决策树的数量
    random_state=7  # 随机种子
)
# 模型训练
model.fit(train_x, train_y)
# 模型预测
pred_test_y = model.predict(test_x)
# 模型评估
print('adaboosting,r2得分：', sm.r2_score(test_y, pred_test_y))

print('*' * 60)

# 利用GBDT模型实现波士顿房价预测

# 定义模型
model = se.GradientBoostingRegressor(
    max_depth=4,  # 子模块模型
    n_estimators=1000,  # 决策树的数量
    min_samples_split=5
)

# 模型训练
model.fit(train_x, train_y)
pred_test_y = model.predict(test_x)

# 模型评估
print('GBDT,r2得分：', sm.r2_score(test_y, pred_test_y))

print('*' * 60)

"""
    利用随机森林实现波士顿房价预测
"""

# 模型定义
model = se.RandomForestRegressor(
    max_depth=10,  # 最大深度
    n_estimators=1000,  # 决策树的数量
    min_samples_split=5,  # 样本最小数量
)

# 模型训练
model.fit(train_x, train_y)
# 模型预测
pred_test_y = model.predict(test_x)

# 模型评估
print('随机森林r2得分', sm.r2_score(test_y, pred_test_y))
print('#' * 60)
"""
    基于GBDT模型实现共享单车投放量预测
"""

# 获取样本数据
data = pd.read_csv('data_test/bike_day.csv')
# print(data.shape)

# 数据处理
# 删除序号，日期，游客使用量以及注册用户使用量这些列
data = data.drop(['instant', 'dteday', 'casual', 'registered'], axis=1)
print(data.shape)

# 整理数据的输入集与输出集
x = data.iloc[:, :-1]  # 输入集是二维数据
y = data.iloc[:, -1]
# print(x.shape,y.shape)

# 划分测试集与训练集
train_x, test_x, train_y, test_y = ms.train_test_split(
    x,  # 输入集
    y,  # 输出集
    test_size=0.1,  # 测试集所占的比例
    random_state=7
)
# 构建模型GBDT
model = se.GradientBoostingRegressor(
    max_depth=6,
    n_estimators=600,
    min_samples_split=8
)

# 模型训练
model.fit(train_x, train_y)
# 模型预测
pred_test_y = model.predict(test_x)
# 模型评估
# r2得分
print(sm.r2_score(test_y, pred_test_y))
# 均方误差
print(sm.mean_squared_error(test_y, pred_test_y))
# 绝对值误差
print(sm.mean_absolute_error(test_y, pred_test_y))
