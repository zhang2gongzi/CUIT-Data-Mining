import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

# 加载数据
data = pd.read_csv('课后作业.csv')

# 提取特征和目标变量
X = data.drop('group', axis=1)
y = data['group']

# 检查是否有缺失值
print("Missing values in each column:")
print(X.isnull().sum())

# 定义数值型和分类型特征
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

# 创建预处理器
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),  # 使用均值填充缺失值
    ('scaler', StandardScaler())  # 标准化
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),  # 填充缺失值
    ('onehot', OneHotEncoder(handle_unknown='ignore'))  # 独热编码
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 创建完整的机器学习管道
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# 划分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 定义超参数网格
param_grid = {
    'classifier__n_estimators': [50, 100, 200],  # 决策树的数量
    'classifier__max_depth': [None, 10, 20, 30],  # 树的最大深度
    'classifier__min_samples_split': [2, 5, 10],  # 内部节点再划分所需最小样本数
    'classifier__min_samples_leaf': [1, 2, 4],  # 叶子节点所需最小样本数
    'classifier__bootstrap': [True, False]  # 是否有放回地抽样
}

# 创建网格搜索对象
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)

# 执行网格搜索
grid_search.fit(X_train, y_train)

# 输出最佳参数
print("Best parameters found: ", grid_search.best_params_)

# 使用最佳参数的模型进行预测
best_model = grid_search.best_estimator_
accuracy = best_model.score(X_test, y_test)
print(f"Model accuracy with best parameters: {accuracy:.2f}")

# 使用5折交叉验证评估最佳模型
cv_scores = cross_val_score(best_model, X, y, cv=5)
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean cross-validation score: {cv_scores.mean():.2f}")

# 获取特征重要性
# 注意：由于我们使用了Pipeline和ColumnTransformer，直接获取特征重要性需要额外处理
# 这里我们假设所有特征都已经经过处理并且可以按顺序访问

# 获取独热编码后的特征名称
# 首先，我们需要拟合并转换训练数据以获取编码后的特征名称
X_train_preprocessed = best_model.named_steps['preprocessor'].fit_transform(X_train)

# 获取数值型特征的名称
numeric_feature_names = numeric_features

# 获取独热编码后的分类型特征名称
if len(categorical_features) > 0:
    categorical_feature_names = list(best_model.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features))
else:
    categorical_feature_names = []

# 合并所有特征名称
feature_names = numeric_feature_names + categorical_feature_names

# 获取特征重要性
importances = best_model.named_steps['classifier'].feature_importances_
indices = (-importances).argsort()

# 打印特征重要性
for i in indices:
    print(f"{feature_names[i]}: {importances[i]:.3f}")

# 可视化特征重要性
plt.figure(figsize=(12, 6))
plt.title("Feature Importances")
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel("Relative Importance")
plt.show()