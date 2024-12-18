import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# 加载数据
data = pd.read_csv('课后作业.csv')

# 显示数据的前几行以检查是否正确加载
print(data.head())

# 提取特征和目标变量
X = data.drop('group', axis=1)
y = data['group']

# 划分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建随机森林分类器模型
model = RandomForestClassifier(random_state=42)

# 训练模型
model.fit(X_train, y_train)

# 评估模型性能
accuracy = model.score(X_test, y_test)
print(f"Model accuracy: {accuracy:.2f}")

# 使用5折交叉验证评估模型
cv_scores = cross_val_score(model, X, y, cv=5)
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean cross-validation score: {cv_scores.mean():.2f}")

# 获取特征重要性
importances = model.feature_importances_
indices = (-importances).argsort()

# 打印特征重要性
for i in indices:
    print(f"{X.columns[i]}: {importances[i]:.3f}")

# 可视化特征重要性
plt.figure(figsize=(12, 6))
plt.title("Feature Importances")
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), X.columns[indices])
plt.xlabel("Relative Importance")
plt.show()