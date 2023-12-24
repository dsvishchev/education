import pandas as pd
import numpy as np

from perceptron import Perceptron

count = 0


if __name__ == '__main__':
    df = pd.read_csv('../data/data.csv')
    df = df.iloc[np.random.permutation(len(df))]
    y = df.iloc[0:100, 4].values
    y = np.where(y == "Iris-setosa", 1, -1)
    X = df.iloc[0:100, [0, 2]].values

    inputSize = X.shape[1]  # количество входных сигналов равно количеству признаков задачи
    hiddenSizes = 10  # задаем число нейронов скрытого (А) слоя
    outputSize = 1 if len(y.shape) else y.shape[1]  # количество выходных сигналов равно количеству классов задачи
    perceptron = Perceptron(inputSize, hiddenSizes, outputSize)

    while not perceptron.is_train_completed():
        count = count + 1
        perceptron.train(X, y)
        y = df.iloc[:, 4].values
        y = np.where(y == "Iris-setosa", 1, -1)
        X = df.iloc[:, [0, 2]].values
        out, hidden_predict = perceptron.predict(X)

        perceptron.error = sum(out - y.reshape(-1, 1))

    print(perceptron.Wout)
    print(f'Error: {perceptron.error}')
    print(f'Count: {count}')