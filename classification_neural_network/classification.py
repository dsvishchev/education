import numpy as np
import pandas as pd
from neural_network import NeuralNetwork
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


if __name__ == '__main__':
    # Загрузка и подготовка данных
    df = pd.read_csv('../data/data.csv')

    df = df.iloc[np.random.permutation(len(df))]
    y = df.iloc[0:100, 4].values
    y = LabelEncoder().fit_transform(y)
    X = df.iloc[0:100, 0:3].values
    y_encoded = np.eye(3)[y]

    # Деление данных на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # Создание и обучение нейронной сети
    input_shape = X_train.shape[1]
    output_shape = y_train.shape[1]

    neural_network = NeuralNetwork(input_shape, 10, output_shape)
    neural_network.train(X_train, y_train, 0.01)

    # Предсказание
    test_predictions = neural_network.predict(X_test)

    # Округление активационных значений до 0 или 1 для классификации
    test_predictions_rounded = np.round(test_predictions)

    # Вычисление точности на обучающей и тестовой выборке
    test_accuracy = np.mean(test_predictions_rounded == y_test)

    print('Точность на тестовой выборке', test_accuracy)