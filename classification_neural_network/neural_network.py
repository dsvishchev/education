import numpy as np


class NeuralNetwork:
    def __init__(self, input_shape, hidden_shape, output_shape):
        # Инициализация весов синапсов скрытого слоя
        self.W_hidden = np.random.uniform(size=(input_shape, hidden_shape))

        # Инициализация весов синапсов выходного слоя
        self.W_output = np.random.uniform(size=(hidden_shape, output_shape))

        # Инициализация максимального количества весов
        self.max_epochs = 1000

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def set_max_epochs(self, value):
        self.max_epochs = value

    def forward_pass(self, X):
        # Прямой проход через скрытый слой
        hidden_layer_output = self.sigmoid(np.dot(X, self.W_hidden))

        # Прямой проход через выходной слой
        output_layer_output = self.sigmoid(np.dot(hidden_layer_output, self.W_output))

        return output_layer_output, hidden_layer_output

    def backward_pass(self, X, y, output, hidden_output, learning_rate):
        # Обратное распространение ошибки на выходном слое
        output_error = y - output
        output_delta = output_error * self.sigmoid_derivative(output)

        # Обратное распространение ошибки на скрытом слое
        hidden_error = np.dot(output_delta, self.W_output.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(hidden_output)

        # Корректировка весов синапсов выходного и скрытого слоя
        self.W_output += learning_rate * np.dot(hidden_output.T, output_delta)

        self.W_hidden += learning_rate * np.dot(X.T, hidden_delta)

    def train(self, X, y, learning_rate):
        len_epochs = len(y)
        print(f'Max epochs {self.max_epochs}')
        while not self.check_is_stop_train(X, y):
            randIndexes = np.random.permutation(len_epochs)
            X = X[randIndexes]
            y = y[randIndexes]
            # Стохастический градиентный спуск
            for j in range(len(X)):
                X_sample = X[j].reshape(1, -1)
                y_sample = y[j].reshape(1, -1)

                output, hidden_output = self.forward_pass(X_sample)
                self.backward_pass(X_sample, y_sample, output, hidden_output, learning_rate)

    def predict(self, X):
        output, _ = self.forward_pass(X)
        return output

    def check_is_stop_train(self, X, Y):
        if self.check_is_accuracy_good(X, Y):
            return True
        if self.check_epochs_count_limited():
            return True
        return False

    def check_is_accuracy_good(self, X, Y):
        # Предсказание
        predictions = self.predict(X)

        # Округление активационных значений до 0 или 1 для классификации
        predictions_rounded = np.round(predictions)

        # Вычисление точности
        accuracy = np.mean(predictions_rounded == Y)

        print(accuracy)
        return accuracy >= 0.95

    def check_epochs_count_limited(self):
        current_epochs = self.max_epochs - 1
        self.set_max_epochs(current_epochs)
        return current_epochs == 0