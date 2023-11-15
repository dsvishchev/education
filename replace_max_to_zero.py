import numpy

"""
Лабораторная работа №1
Создать массив со случайными значениями и максимальный элемент заменить на 0
"""
if __name__ == '__main__':
    array = numpy.random.randint(100, size=10)
    print(array)
    array[numpy.argmax(array)] = 0
    print(array)