import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.datasets import load_breast_cancer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 1. 载入数据
loaded_data = load_breast_cancer()
X = loaded_data.data
y = loaded_data.target

# 2. 将X正则化
X = preprocessing.scale(X)

# 3. 数据重塑
X = X.reshape(X.shape[0], X.shape[1], 1)  # 重塑为 (样本数, 特征数, 通道数)

# 4. 划分训练集、验证集和测试集
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=120)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=120)

# 设定交叉验证参数
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=120)
accuracies = []
losses = []

for train_index, val_index in kf.split(X, y):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]

    # 构建模型
    model = keras.Sequential([
        keras.layers.Conv1D(64, 3, activation='relu', input_shape=(X.shape[1], 1)),
        keras.layers.MaxPooling1D(2),
        keras.layers.Conv1D(128, 3, activation='relu'),
        keras.layers.MaxPooling1D(2),
        keras.layers.Conv1D(128, 3, activation='relu'),
        keras.layers.MaxPooling1D(2),
        keras.layers.Dropout(0.5),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(2, activation='softmax')
    ])

    # 编译模型
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # 7. 设置早停和学习率调度器
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True)  # 修改了监控参数和耐心值
    lr_scheduler = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.1, patience=5)  # 修改了监控参数和学习率调整因子

    # 训练模型
    history = model.fit(X_train, y_train, epochs=50, validation_data=(X_val, y_val), callbacks=[early_stopping, lr_scheduler], verbose=0)

    # 记录每个折叠的精度和损失
    val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
    accuracies.append(val_accuracy)
    losses.append(val_loss)

# 输出交叉验证结果
print(f"交叉验证精度: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
print(f"交叉验证损失: {np.mean(losses):.4f} ± {np.std(losses):.4f}")

# 9. 绘制训练集和验证集精度和损失
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

# 10. 在测试集上评估模型
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"测试集损失: {test_loss:.4f}")
print(f"测试集精度: {test_accuracy:.4f}")

# 预测
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# 计算混淆矩阵
cm = confusion_matrix(y_test, y_pred_classes)

# 绘制混淆矩阵
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['类0', '类1'], yticklabels=['类0', '类1'])
plt.ylabel('真实标签')
plt.xlabel('预测标签')
plt.title('混淆矩阵')
plt.show()