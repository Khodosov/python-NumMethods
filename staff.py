import math
import numpy as np


def norm_of_Matrix(matrix, len):
    max = 0
    for i in range(0, len):
        current = 0
        for j in range(0, len):
            current += abs(matrix[i, j])
        if current > max:
            max = current
    return max


def methodYakobi_Zeidel(matrix, len, b, currency=1e-12):  # пусть точносмть 10^-12
    c_matrix = matrix.copy()
    b_c = b.copy()
    for i in range(0, len):
        factor = c_matrix[i, i]  # нулей нет
        c_matrix[i, i] = 0
        b_c[0, i] /= factor  # меняем столбец b
        for j in range(0, len):
            c_matrix[i, j] /= -factor  # строим B
    norm = norm_of_Matrix(c_matrix, len)  # норма матрицы
    if norm < 1:  # условие сходимости
        k = 1  # счет итераций (первая вне цикла)
        xk = b_c.T
        xn = c_matrix.dot(xk) + b_c.T  # начинается метод якоби
        apr_k = math.floor(math.log(currency * (1 - norm) / max(abs(xn - xk)), norm))
        print("Априорная оценка: ", apr_k, "\n")
        while max(abs(xn - xk)) > (1 - norm) / norm * currency:  # МЕТОД ЯКОБИ
            xk = xn  # xk - предыдущий результат, xn - текущий
            xn = c_matrix.dot(xk) + b_c.T
            k += 1  # Подсчет числа итераций
        print("Реальное число итераций методом Якоби :", k, "\n")
        result1 = xn
        # пошел метод Зейделя
        k = 0  # счет итераций
        xk = b_c.T.copy()  # xk - текущий результат, xn - предыдущий
        xn = xk.copy()
        while True:  # чтоб не делать итерацию вне цикла
            xn = xk.copy()
            for i in range(0, len):
                xk[i] = c_matrix[i].dot(xk) + b_c[0, i]
            k += 1
            if max(abs(xn - xk)) < currency:
                break
        print("Число итераций методом Зейделя: ", k, "\n")
        result2 = xk
        return [result1, result2]
    else:
        print("Для данной матрицы метод не применим")  # Норма >=1


# любая
matrix1 = np.array([[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 10]], dtype=np.float64)

# чтоб на главной диагонали элементы были больше суммы остальных элемнентов
matrix = np.array([[220, 76, 52, 23],
                   [74, 227, 92, 55],
                   [4, 40, 190, 83],
                   [12, 0, 54, 74]], dtype=np.float64)
length = len(matrix)
x = methodYakobi_Zeidel(matrix, length, np.array([[15, 82, 41, 6]], dtype=np.float64))
print("X методом Якоби :", x[0], "\n")
print("Точность X методом Якоби: ", matrix.dot(x[0]) - np.array([[15, 82, 41, 6]]).T, "\n")
print("X методом Зейделя :", x[1], "\n")
print("Точность X методом Зейделя: ", matrix.dot(x[1]) - np.array([[15, 82, 41, 6]]).T, "\n")
