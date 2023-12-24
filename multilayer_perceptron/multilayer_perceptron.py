import torch
import torch.nn as nn
from time import time


class MultiLayeredPerceptron(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, function, lr):
        nn.Module.__init__(self)
        self.train_time = 0
        self.error_count = 0
        self.lossFn = nn.MSELoss()
        self.linear = nn.Linear(3, 2)
        self.optimizer = torch.optim.SGD(self.linear.parameters(), lr=lr)
        # nn.Sequential - контейнер модулей
        # он последовательно объединяет и позволяет запускать их одновременно
        linear = nn.Linear(in_size, hidden_size)
        print(linear.weight)
        self.layers = nn.Sequential(linear,  # слой линейных сумматоров
                                    nn.ReLU(),  # функция активации
                                    nn.Linear(hidden_size, hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(hidden_size, hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(hidden_size, out_size),
                                    nn.ReLU(),
                                    )
        print(self.layers[0])

    def forward(self, x):
        """
        Прямой проход
        :param x:
        :return:
        """
        return self.layers(x)

    def train_num_iter(self, x, y, num_iter, relu=True):
        loss = None
        start_time = time()
        for i in range(0, num_iter):
            # Делаем предсказание
            pred = self.forward(x)
            #print(pred)
            # Вычисляем ошибку
            loss = self.lossFn(pred, y)
            # Обратное распределение
            loss.backward()
            # Шаг оптимизации
            self.optimizer.step()
            if i % 100 == 0 and relu:
                print('Ошибка на ' + str(i + 1) + ' итерации: ', loss.item())
        self.train_time = time() - start_time
        return loss.item()