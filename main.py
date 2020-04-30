import numpy as np

# В процессе выполнения программы, порой могут возникать погрешности вычисления, отсюда некоторые неточности в
# сравнении, например, PA = LU

# Выбран поиск опорного элемента по столбцу => матрица Q(matrix_q) будет единичной, det(Q) = 1. Все функции работают
# без использования матрицы Q

# matrix_q - матрица перестановки
# matrix_Q - матрица Q из QR разложения

# Здесь мы зададим размерность матрицы
size = 3


def matrix_generation(n):
    return np.random.randint(0, 10, (n, n)).astype(np.float32)


def vector_generation(n):
    return np.random.randint(0, 10, (n, 1)).astype(np.float32)


def swap(matrix, k, n):
    matr = np.copy(matrix)
    matr[n, :] = matr[k, :]
    matr[k, :] = matrix[n, :]
    return matr


def det_U(matrix, size, swaps):
    det = (-1) ** swaps
    for i in range(size):
        det *= matrix[i, i]
    return det


def decompose_LU(matrix, n):
    matrix_c = np.copy(matrix)

    swaps = 0
    matrix_p = np.eye(n)
    matrix_q = np.eye(n)
    matrix_l = np.zeros((n, n))

    for i in range(n):
        pivot = -1
        pivotValue = np.fabs(matrix_c[i, i])
        for row in range(i + 1, n):
            if np.fabs(matrix_c[row, i]) > pivotValue:
                pivotValue = np.fabs(matrix_c[row, i])
                pivot = row
        if pivot != -1:
            swaps += 1
            matrix_c = swap(matrix_c, pivot, i)
            matrix_p = swap(matrix_p, pivot, i)
            matrix_l = swap(matrix_l, pivot, i)
        matrix_l[i, i] = 1
        for j in range(i + 1, n):
            coeff = matrix_c[j, i] / matrix_c[i, i]
            matrix_l[j, i] = coeff
            for k in range(i, n):
                matrix_c[j, k] -= coeff * matrix_c[i, k]

    matrix_u = matrix_c
    return matrix_u, matrix_l, matrix_p, matrix_q, swaps


def system_solution_LU(matrix_l, matrix_u, n, b, matrix_p):
    # Ax = b => PAx = LUx = Pb
    res_x = np.zeros((n, 1))
    b = matrix_p.dot(b)
    y = []
    x = []
    for i in range(n):
        k = b[i]
        for j in range(n)[0:i]:
            k -= matrix_l[i, j] * y[j]
        y.append(k)
    for i in range(n):
        k = y[n - (i + 1)]
        for j in range(n)[0:i]:
            k -= matrix_u[n - (i + 1), n - (j + 1)] * x[j]
        x.append(k / matrix_u[n - (i + 1), n - (i + 1)])
    x.reverse()
    for i in range(n):
        res_x[i, 0] = x[i]
    return res_x


def inverse(matrix_l, matrix_u, matrix_p, n):
    matrix_inv = []
    eye = np.eye(n)
    for i in range(n):
        matrix_inv.append(system_solution_LU(matrix_l, matrix_u, n, eye[:, i], matrix_p))
    return np.transpose(matrix_inv)


def decompose_QR(matrix, n):
    matrix_q = np.zeros((n, n))
    matrix_r = np.zeros((n, n))
    for j in range(n):
        matrix_q[:, j] = matrix[:, j]
        for i in range(j - 2):
            matrix_r[i, j] = np.transpose(matrix_q[:, i]).dot(matrix[:, j])
            matrix_q[:, j] = matrix_q[:, j] - matrix_r[j, i] * matrix_q[:, i]
        matrix_r[j, j] = np.linalg.norm(matrix_q[:, j])
        if matrix_r[j, j] == 0:
            pass
        else:
            matrix_q[:, j] = matrix_q[:, j] / matrix_r[j, j]

    return matrix_q, matrix_r


def system_solution_QR(matrix_r, matrix_Q, b):
    # A = QR
    # Ax = b
    # QRx = b
    # Q^(-1)*QRx = Q^(-1)b
    # ERx = Q^(-1)b
    # Rx = Q^(-1)b
    return np.linalg.solve(matrix_r, (np.linalg.inv(matrix_Q).dot(b)))


def seidel(matrix, b, accuracy, n):
    # matrix_D - матрица на главной диагонале которой, распологаются элементы главной диагонали исходной
    # матрицы (matrix)
    # matrix_L - матрица, которая содержит элементы исходной, стоящие под главной диагональю
    # matrix_U - матрица, которая содержит элементы исходной, стоящие над главной диагональю

    matrix_D = np.zeros((n, n))
    for i in range(n):
        matrix_D[i, i] = matrix[i, i]
    matrix_L = np.tril(matrix) - matrix_D
    matrix_U = np.triu(matrix) - matrix_D
    x_previous = np.zeros((n, 1))
    # Одну итеррацию вне цикла, чтобы запустить сам цикл
    x_current = ((np.linalg.inv(matrix_L + matrix_D)).dot(-1 * matrix_U.dot(x_previous))) + (
        np.linalg.inv(matrix_L + matrix_D).dot(b))
    norma = 10000
    while norma > accuracy:
        x_current = ((np.linalg.inv(matrix_L + matrix_D)).dot(-1 * matrix_U.dot(x_previous))) + (
            np.linalg.inv(matrix_L + matrix_D).dot(b))
        norma = np.linalg.norm(x_current - x_previous)
        x_previous = np.copy(x_current)

    return x_current


# =====
# №1
# =====
matrix = matrix_generation(size)
print("==== Matrix A ====================================================" + "\n", matrix)
matrix_u, matrix_l, matrix_p, matrix_q, swaps = decompose_LU(matrix, size)
print("==== Matrix L ====================================================" + "\n", matrix_l)
print("==== Matrix U ====================================================" + "\n", matrix_u)
print("==== Matrix Q ====================================================" + "\n", matrix_q)
print("==== Matrix P ====================================================" + "\n", matrix_p)
print("==== Matrix PA ===================================================" + "\n", matrix_p.dot(matrix))
print("==== Matrix LU ===================================================" + "\n", matrix_l.dot(matrix_u))
print("==== det A =======================================================" + "\n", np.linalg.det(matrix))
print("==== Произведение диагональных элементов U =======================" + "\n", det_U(matrix_u, size, swaps))
b = vector_generation(size)
x = system_solution_LU(matrix_l, matrix_u, size, b, matrix_p)
print("==== A, b ========================================================" + "\n", matrix, "\n", b)
print("==== X ===========================================================" + "\n", x)
print("==== Ax ==========================================================" + "\n", matrix.dot(x))
matrix_inv = inverse(matrix_l, matrix_u, matrix_p, size)
print("==== A^(-1) ======================================================" + "\n", np.linalg.inv(matrix))
print("==== U^(-1) * L^(-1) * P =========================================" + "\n",
      (np.linalg.inv(matrix_u).dot(np.linalg.inv(matrix_l))).dot(matrix_p))
print("==== A^(-1) Получено с помощью LU ================================" + "\n", matrix_inv)
print("==== A * A^(-1) ==================================================" + "\n", matrix.dot(matrix_inv))
print("==== Число обусловленности А =====================================" + "\n",
      np.linalg.norm(matrix_inv) * np.linalg.norm(matrix))
# =====
# №2
# =====

# =====
# №3
# =====
matrix_Q, matrix_r = decompose_QR(matrix, size)
print("==== Q ===========================================================" + "\n", matrix_Q)
print("==== R ===========================================================" + "\n", matrix_r)
print("==== Q * R =======================================================" + "\n", matrix_Q.dot(matrix_r))
print("==== x ===========================================================" + "\n",
      system_solution_QR(matrix_r, matrix_Q, b))

# =====
# № 4
# =====
accuracy = 1e-12
print("==== Метод Зейделя ===============================================" + "\n", seidel(matrix, b, accuracy, size))
