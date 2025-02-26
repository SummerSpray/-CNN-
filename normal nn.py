# -*- coding: UTF-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.datasets import load_breast_cancer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 1. 载入数据
loaded_data = load_breast_cancer()
X = loaded_data.data
y = loaded_data.target

# 2. 将X正则化
X = preprocessing.scale(X)

# 3. 划分训练集、验证集和测试集
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=120)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=120)

# 4. 构建普通神经网络模型
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(X.shape[1],)),  # 输入层
    keras.layers.Dense(64, activation='relu'),  # 隐藏层
    keras.layers.Dense(2, activation='softmax')  # 输出层
])

# 5. 编译模型
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 6. 训练模型
history = model.fit(X_train, y_train, epochs=50, validation_data=(X_val, y_val))

# 7. 绘制训练集和验证集精度和损失
# 精度
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='训练集精度')
plt.plot(history.history['val_accuracy'], label='验证集精度')
plt.title('训练集和验证集精度')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# 损失
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='训练集损失')
plt.plot(history.history['val_loss'], label='验证集损失')
plt.title('训练集和验证集损失')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# 8. 在测试集上评估模型
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"测试集损失: {test_loss:.4f}")
print(f"测试集精度: {test_accuracy:.4f}")