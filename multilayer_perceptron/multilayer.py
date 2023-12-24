import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from multilayer_perceptron import MultiLayeredPerceptron
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

if __name__ == '__main__':
    df = pd.read_csv('../data/data.csv')

    y = df.iloc[0:100, 4].values
    y = np.where(y == "Iris-setosa", 1, -1)
    y = LabelEncoder().fit_transform(y)
    X = df.iloc[0:100, 0:2].values
    Y = np.eye(2)[y]

    # Деление данных на обучающую и тестовую выборки
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    inputSize = X.shape[1]  # количество входных сигналов равно количеству признаков задачи
    hiddenSizes = 50  # задаем число нейронов скрытого слоя
    outputSize = Y.shape[1] if len(Y.shape) else 1  # количество выходных сигналов равно количеству классов задачи

    net_relu = MultiLayeredPerceptron(inputSize, hiddenSizes, outputSize, nn.ReLU(), 0.1)
    net_sigmoid = MultiLayeredPerceptron(inputSize, hiddenSizes, outputSize, nn.Sigmoid(), 0.01)
    num_iter = 5000

    net_relu.train_num_iter(torch.from_numpy(X.astype(np.float32)), torch.from_numpy(Y.astype(np.float32)), num_iter)
    net_sigmoid.train_num_iter(torch.from_numpy(X.astype(np.float32)), torch.from_numpy(Y.astype(np.float32)), num_iter, False)

    print(f"Время обучения с использванием функции активации ReLU: {net_relu.train_time}")
    print(f"Время обучения с использванием функции активации Sigmoid: {net_sigmoid.train_time}")

    pred = net_relu.forward(torch.from_numpy(X_test.astype(np.float32))).detach().numpy()
    a = pred > 0.5
    b = a - Y_test
    err = sum(abs((pred > 0.5) - Y_test))
    print(err)

    pred = net_sigmoid.forward(torch.from_numpy(X_test.astype(np.float32))).detach().numpy()
    a = pred > 0.5
    b = a - Y_test
    err = sum(abs((pred > 0.5) - Y_test))
    print(err)