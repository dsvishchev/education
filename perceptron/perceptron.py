import numpy as np


class Perceptron:
    def __init__(self, inputSize, hiddenSizes, outputSize):
        self.error = None
        self.w_matrix_map = {}

        self.Win = np.zeros((1 + inputSize, hiddenSizes))
        self.Win[0, :] = np.random.randint(0, 3, size=hiddenSizes)
        self.Win[1:, :] = np.random.randint(-1, 2, size=(inputSize, hiddenSizes))

        self.Wout = np.random.randint(0, 2, size=(1 + hiddenSizes, outputSize)).astype(np.float64)

    def predict(self, X):
        hidden_out = np.where((np.dot(X, self.Win[1:, :]) + self.Win[0, :]) >= 0.0, 1, -1).astype(np.float64)
        prediction = np.where((np.dot(hidden_out, self.Wout[1:, :]) + self.Wout[0, :]) >= 0.0, 1, -1).astype(np.float64)
        return prediction, hidden_out

    def train(self, X, Y, etta=0.01):
        for x, y in zip(X, Y):
            pr, hidden_out = self.predict(x)
            self.Wout[1:] += (hidden_out * ((y - pr) * etta)).reshape(-1, 1)
            self.Wout[0] += etta * (y - pr)
            self.save_weight_matrix(self.Wout)

    def save_weight_matrix(self, weight_matrix):
        weight_matrix_key = ''.join(str(w) for w in list(weight_matrix))
        keys = self.w_matrix_map.keys()
        r = list(filter(lambda k: k == weight_matrix_key, keys))
        if len(r) != 0:
            self.w_matrix_map[weight_matrix_key] += 1
        else:
            self.w_matrix_map[weight_matrix_key] = 1

    def is_train_completed(self):
        """
        Пока существуют ошибки и вектор весов не повторяется результат функции будет равен False
        перцептрон :param perceptron:
        булево значение для остановки обучения перцептрона :return:
        """

        if self.error == 0:
            return True
        else:
            values = list(self.w_matrix_map.values())
            if any([v > 2 for v in values]):
                return True
            else:
                return False