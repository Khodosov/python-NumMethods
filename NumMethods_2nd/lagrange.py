import numpy as np

X_array = np.array([0.43, 0.48, 0.55, 0.62, 0.70, 0.65, 0.67, 0.69, 0.71, 0.74], dtype=float)
Y_array = np.array([1.63597, 1.73234, 1.87686, 2.03345, 2.22846, 2.35973, 2.52168, 2.80467, 2.98146, 3.14629],
                   dtype=float)


def lagrange(x, y, p):
    result = 0
    for j in range(len(y)):
        p1 = 1
        p2 = 1
        for i in range(len(x)):
            if i == j:
                p1 = p1 * 1
                p2 = p2 * 1
            else:
                p1 = p1 * (p - x[i])
                p2 = p2 * (x[j] - x[i])
        result = result + y[j] * p1 / p2
    return result


print("=== Вариант № 2 точка 0.512 ===")
print("=== f(0.512): ==================")
print("=== " + str(lagrange(X_array, Y_array, 0.512)) + " ==========")
