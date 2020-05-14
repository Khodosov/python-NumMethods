import time

import numpy as np
from sympy import *

# Для выполнения задания необходимо восполльзоваться методами, реализованными в предыдущем задании.


size = 10
accuracy = 0.0001


def swap_columns2(your_list, pos1, pos2):
    for item in your_list:
        item[pos1], item[pos2] = item[pos2], item[pos1]


def swap_rows2(input_array, row1, row2):
    temp = np.copy(input_array[row1][:])
    input_array[row1][:] = input_array[row2][:]
    input_array[row2][:] = temp


def decompose_PAQ_LU(matrix, n):
    swaps = 0
    matrix_c = np.copy(matrix)
    matrix_p = np.eye(n)
    matrix_q = np.eye(n)

    for i in range(n):
        pivotValue = 0
        pivot1 = -1
        pivot2 = -1
        for row in range(n)[i:n]:
            for column in range(n)[i:n]:
                if np.fabs(matrix_c[row][column]) > pivotValue:
                    pivotValue = np.fabs(matrix_c[row][column])
                    pivot1 = row
                    pivot2 = column
        if pivotValue != 0:
            if pivot1 != i:
                swaps += 1
                swap_rows2(matrix_p, pivot1, i)
                swap_rows2(matrix_c, pivot1, i)
            if pivot2 != i:
                swaps += 1
                swap_columns2(matrix_q, pivot2, i)
                swap_columns2(matrix_c, pivot2, i)
            for j in range(n)[i + 1:n]:
                matrix_c[j][i] /= matrix_c[i][i]
                for s in range(n)[i + 1:n]:
                    matrix_c[j][s] -= matrix_c[j][i] * matrix_c[i][s]

    matrix_l = np.tril(np.array(matrix_c), -1) + np.identity(n)
    matrix_u = np.triu(np.array(matrix_c), 0)
    return matrix_l, matrix_u, matrix_p, matrix_q, swaps


def rang(matrix, n):
    rank = n
    for row in range(n):
        if zeros(matrix[n - 1 - row], n):
            rank -= 1
        else:
            break
    return rank


def zeros(row, n):
    for i in range(n):
        if abs(row[i]) - 0.0000001 > 0:
            return False
    return True


# Для совместной
def system_solution_2(matrix_l, matrix_u, n, b, matrix_p, matrix_q):
    operations_number = 0
    res_x = np.zeros((n, 1))
    b = matrix_p.dot(b)
    y = []
    x = []
    for i in range(n):
        k = b[i]
        for j in range(n)[0:i]:
            k -= matrix_l[i, j] * y[j]
            operations_number += 2
        y.append(k)
    for i in range(n):
        k = y[n - (i + 1)]
        for j in range(n)[0:i]:
            k -= matrix_u[n - (i + 1), n - (j + 1)] * x[j]
            operations_number += 2
        x.append(k / matrix_u[n - (i + 1), n - (i + 1)])
        operations_number += 1
    x.reverse()
    for i in range(n):
        res_x[i, 0] = x[i]
    operations_number += 2 * n * n
    return matrix_q.dot(res_x), operations_number


# Для решения системы с вырожденной матрицей
def solve_3(L, U, P, Q, b, rank, n):
    operations_number = 0
    cU = np.copy(U)
    g = np.linalg.inv(L).dot(P).dot(b)
    operations_number += 4 * n * n
    y = np.zeros(n)
    for i in range(rank)[::-1]:
        g[i] = g[i] / cU[i, i]
        cU[i, :] /= cU[i, i]
        operations_number += n + 1
        for j in range(i):
            g[j] -= g[i] * cU[j, i]
            operations_number += 2
            cU[j, :] -= cU[i, :] * cU[j, i]
            operations_number += 2 + 2 * n
        y[i] = g[i]
    x = Q.dot(y)
    x_res = np.zeros((n, 1))
    operations_number += n * n
    for i in range(n):
        x_res[i, 0] = x[i]
    return x_res, operations_number


# Далее новый код =====================================================================================================


def func(a):
    res = []
    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10 = symbols('x1 x2 x3 x4 x5 x6 x7 x8 x9 x10')
    Fs = Matrix([cos(x2 * x1) - exp(-3 * x3) + x4 * x5 ** 2 - x6 - sinh(
        2 * x8) * x9 + 2 * x10 + 2.000433974165385440,
                 sin(x2 * x1) + x3 * x9 * x7 - exp(-x10 + x6) + 3 * x5 ** 2 - x6 * (x8 + 1) + 10.886272036407019994,
                 x1 - x2 + x3 - x4 + x5 - x6 + x7 - x8 + x9 - x10 - 3.1361904761904761904,
                 2 * cos(-x9 + x4) + x5 / (x3 + x1) - sin(x2 ** 2) + cos(
                     x7 * x10) ** 2 - x8 - 0.1707472705022304757,
                 sin(x5) + 2 * x8 * (x3 + x1) - exp(-x7 * (-x10 + x6)) + 2 * cos(x2) - 1.0 / (
                         -x9 + x4) - 0.3685896273101277862,
                 exp(x1 - x4 - x9) + x5 ** 2 / x8 + cos(3 * x10 * x2) / 2 - x6 * x3 + 2.0491086016771875115,
                 x2 ** 3 * x7 - sin(x10 / x5 + x8) + (x1 - x6) * cos(x4) + x3 - 0.7380430076202798014,
                 x5 * (x1 - 2 * x6) ** 2 - 2 * sin(-x9 + x3) + 0.15e1 * x4 - exp(
                     x2 * x7 + x10) + 3.5668321989693809040,
                 7 / x6 + exp(x5 + x4) - 2 * x2 * x8 * x10 * x7 + 3 * x9 - 3 * x1 - 8.4394734508383257499,
                 x10 * x1 + x9 * x2 - x8 * x3 + sin(x4 + x5 + x6) * x7 - 0.78238095238095238096])
    for f in Fs:
        res.append(
            float(
                f.subs([(x1, a[0]), (x2, a[1]), (x3, a[2]), (x4, a[3]), (x5, a[4]), (x6, a[5]), (x7, a[6]), (x8, a[7]),
                        (x9, a[8]), (x10, a[9])])))
    return np.array(res)


def jacobian(a):
    res = np.zeros((10, 10))
    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10 = symbols('x1 x2 x3 x4 x5 x6 x7 x8 x9 x10')
    xs = [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10]
    Fs = Matrix([cos(x2 * x1) - exp(-3 * x3) + x4 * x5 ** 2 - x6 - sinh(2 * x8) * x9 + 2 * x10 + 2.000433974165385440,
                 sin(x2 * x1) + x3 * x9 * x7 - exp(-x10 + x6) + 3 * x5 ** 2 - x6 * (x8 + 1) + 10.886272036407019994,
                 x1 - x2 + x3 - x4 + x5 - x6 + x7 - x8 + x9 - x10 - 3.1361904761904761904,
                 2 * cos(-x9 + x4) + x5 / (x3 + x1) - sin(x2 ** 2) + cos(x7 * x10) ** 2 - x8 - 0.1707472705022304757,
                 sin(x5) + 2 * x8 * (x3 + x1) - exp(-x7 * (-x10 + x6)) + 2 * cos(x2) - 1.0 / (-x9 + x4) - 0.3685896273101277862,
                 exp(x1 - x4 - x9) + x5 ** 2 / x8 + cos(3 * x10 * x2) / 2 - x6 * x3 + 2.0491086016771875115,
                 x2 ** 3 * x7 - sin(x10 / x5 + x8) + (x1 - x6) * cos(x4) + x3 - 0.7380430076202798014,
                 x5 * (x1 - 2 * x6) ** 2 - 2 * sin(-x9 + x3) + 0.15e1 * x4 - exp(x2 * x7 + x10) + 3.5668321989693809040,
                 7 / x6 + exp(x5 + x4) - 2 * x2 * x8 * x10 * x7 + 3 * x9 - 3 * x1 - 8.4394734508383257499,
                 x10 * x1 + x9 * x2 - x8 * x3 + sin(x4 + x5 + x6) * x7 - 0.78238095238095238096])
    for i in range(10):
        for j in range(10):
            res[i, j] = float(diff(Fs[i], xs[j]).subs([(x1, a[0]), (x2, a[1]), (x3, a[2]), (x4, a[3]), (x5, a[4]), (x6, a[5]), (x7, a[6]), (x8, a[7]), (x9, a[8]), (x10, a[9])]))
    return res


def SLAE_solve(matrix, _b, n):
    L, U, P, Q, swaps = decompose_PAQ_LU(matrix, n)
    # swaps нам не нужны
    rank = rang(U, n)
    Ab = np.zeros((n, n + 1))
    x = []
    count = 0
    for i in range(n):
        Ab[:, i] = np.copy(matrix[:, i])
    Ab[:, n] = np.copy(_b)
    if np.linalg.matrix_rank(Ab) == rank:
        if rank == n:
            x, count = system_solution_2(L, U, n, _b, P, Q)
        else:
            x, count = solve_3(L, U, P, Q, _b, rank, n)
    return x, count


def newton_method(f, j, x0, n, accuracy):
    x_cur = x0
    iter_num = 0
    operation_num = 0
    t = time.time()
    norm = 1000
    while norm > accuracy:
        iter_num += 1
        apr_x, ops = SLAE_solve(j(x_cur), -f(x_cur), n)
        operation_num += ops
        norm = np.linalg.norm(apr_x, np.inf)
        x_cur = apr_x[:,0].transpose() + x_cur
    duration = time.time() - t
    return x_cur, iter_num, operation_num, duration


def modified_newton_method(f, j, x0, n, accuracy):
    x_cur = x0
    iter_num = 0
    operation_num = 0
    t = time.time()
    norm = 1000
    matrix = j(x_cur)
    L, U, P, Q, swaps = decompose_PAQ_LU(matrix, n)
    # swaps нам не нужны
    while norm > accuracy:
        _b = -f(x_cur)
        rank = rang(U, n)
        Ab = np.zeros((n, n + 1))
        apr_x = []
        counter = 0
        for i in range(n):
            Ab[:, i] = np.copy(matrix[:, i])
        Ab[:, n] = np.copy(_b)
        if np.linalg.matrix_rank(Ab) == rank:
            if rank == n:
                apr_x, counter = system_solution_2(L, U, n, _b,  P, Q)
            else:
                apr_x, counter = solve_3(L, U, P, Q, _b, rank, n)
        operation_num += counter
        iter_num += 1
        norm = np.linalg.norm(apr_x, np.inf)
        x_cur = apr_x[:,0].transpose() + x_cur
    duration = time.time() - t
    return x_cur, iter_num, operation_num, duration


def newton_method_with_change(f, j, x0, k, n, accuracy):
    x_cur = x0
    iter_num = 0
    operation_num = 0
    t = time.time()
    for _ in range(k):
        iter_num += 1
        apr_x, ops = SLAE_solve(j(x_cur), -f(x_cur), n)
        operation_num += ops
        x_cur = apr_x[:,0].transpose() + x_cur
    x_cur, itn, opn, _ = modified_newton_method(f, j, x_cur, n, accuracy)
    duration = time.time() - t
    return x_cur, iter_num + itn, operation_num + opn, duration


def newton_method_with_period(f, j, x0, p, accuracy, n):
    x_cur = x0
    iter_num = 0
    operation_num = 0
    t = time.time()
    norm = 1000
    matrix = j(x_cur)
    L, U, P, Q, swaps = decompose_PAQ_LU(matrix, n)
    # swaps нам не нужны
    period = 0
    while norm > accuracy:
        period += 1
        if period == p:
            matrix = j(x_cur)
            L, U, P, Q, swaps = decompose_PAQ_LU(matrix, n)
            # swaps нам не нужны
            period = 0
        iter_num += 1
        _b = -f(x_cur)
        rank = rang(U, n)
        Ab = np.zeros((n, n + 1))
        apr_x = []
        count = 0
        for i in range(n):
            Ab[:, i] = np.copy(matrix[:, i])
        Ab[:, n] = np.copy(_b)
        if np.linalg.matrix_rank(Ab) == rank:
            if rank == n:
                apr_x, count = system_solution_2(L, U, n, _b, P, Q)
            else:
                apr_x, count = solve_3(L, U, P, Q, _b, rank, n)
        operation_num += count
        norm = np.linalg.norm(apr_x, np.inf)
        x_cur = apr_x[:,0].transpose() + x_cur
    duration = time.time() - t
    return x_cur, iter_num, operation_num, duration


x_0 = np.array([0.5, 0.5, 1.5, -1, -0.5, 1.5, 0.5, -0.5, 1.5, -1.5])
print("==== X_o =========================================================================================")
print(str(x_0))
x, i, u, d = newton_method(func, jacobian, x_0, size, accuracy)
print("==== Метод Ньюьона ===============================================================================")
print("==== X ===========================================================================================" + "\n", x)
print("==== Количество операций =========================================================================" + "\n", u)
print("==== Количество итераций =========================================================================" + "\n", i)
print("==== Время =======================================================================================" + "\n", d)
x, i, u, d = modified_newton_method(func, jacobian, x_0, size, accuracy)
print("==== Модифицированный Метод Ньюьона ==============================================================")
print("==== X ===========================================================================================" + "\n", x)
print("==== Количество операций =========================================================================" + "\n", u)
print("==== Количество итераций =========================================================================" + "\n", i)
print("==== Время =======================================================================================" + "\n", d)
k = 2
x, i, u, d = newton_method_with_change(func, jacobian, x_0, size, accuracy, k)
print("==== Метод Ньюьона с переходом на Модифицированный Метод Ньюьона =================================")
print("==== X ===========================================================================================" + "\n", x)
print("==== Количество операций =========================================================================" + "\n", u)
print("==== Количество итераций =========================================================================" + "\n", i)
print("==== Время =======================================================================================" + "\n", d)
