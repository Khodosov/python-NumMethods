# Вариант № 17

import numpy as np

X_array = np.array([1.340, 1.345, 1.350, 1.355, 1.360, 1.365, 1.370, 1.375, 1.380, 1.385, 1.390, 1.395], dtype=float)
Y_array = np.array(
    [4.25562, 4.35325, 4.45522, 4.56184, 4.67344, 4.79038, 4.91306, 5.04192, 5.17744, 5.32016, 5.47069, 5.62968],
    dtype=float)

x1 = 1.3463
x2 = 1.3868
x3 = 1.335
x4 = 1.3990

x_target = np.array([x1, x2, x3, x4], dtype=float)

dif_array = np.zeros(len(X_array) - 1, dtype=float)
buf_array = np.zeros(len(Y_array), dtype=float)
n = len(X_array)


def finiteDifference(Y_array):
    for i in range(n - 1):
        if i == 0:
            for j in range(n - 1):
                buf_array[j] = Y_array[j + 1] - Y_array[j]
            dif_array[i] = buf_array[0]
            continue
        for j in range(n - 1):
            buf_array[j] = buf_array[j + 1] - buf_array[j]
        dif_array[i] = buf_array[0]
    print("Конечные разности:")
    for dif in dif_array:
        print(dif)
    return dif_array


def newton(X_array, Y_array, x_target_test, dif_array):
    print("Приближения:")
    for x in x_target_test:
        aprox = Y_array[0]
        q = (x - X_array[0]) / (X_array[1] - X_array[0])
        for j in range(n - 1):
            k = 1.0
            for p in range(j + 1):
                k = k * (q - float(p)) / (float(p) + 1)
            aprox += k * dif_array[j]
        print("f("+ str(x) + ") = ", aprox)


dif_for_func = finiteDifference(Y_array)
newton(X_array, Y_array, x_target, dif_for_func)
