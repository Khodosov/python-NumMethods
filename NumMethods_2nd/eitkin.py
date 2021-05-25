# Вариант № 17

import numpy as np

X_array = np.array([0.62, 0.67, 0.74, 0.80, 0.87, 0.96, 0.99], dtype=float)
Y_array = np.array([0.537944, 0.511709, 0.477114, 0.449329, 0.418952, 0.382893, 0.371577], dtype=float)
x_target = 0.683


def eitkin(X_array, Y_array, x_target):
    n = len(X_array)
    P_array = np.zeros(n, dtype=float)
    for i in range(n - 1):
        if i == 0:
            for j in range(n - 1):
                P_array[j] = 1 / (X_array[j + 1] - X_array[j]) * (
                        Y_array[j] * (X_array[j + 1] - x_target) - Y_array[j + 1] * (X_array[j] - x_target))
            continue
        for j in range(n - 1 - i):
            P_array[j] = 1 / (X_array[j + 1 + i] - X_array[j]) * (
                    P_array[j] * (X_array[j + 1 + i] - x_target) - P_array[j + 1] * (X_array[j] - x_target))

    return P_array[0]


print("f(" + str(x_target) + ") = ", eitkin(X_array, Y_array, x_target))
