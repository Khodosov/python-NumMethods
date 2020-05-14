import time
from copy import deepcopy

import numpy as np
from sympy import *

# Время выполнения программы очень большое из-за использования SymPy при подсчете значения функций и якобиана.
SIZE = 10

# Точность для всех вариаций метода Ньютона. Очень сильно влияет на длительность, особенно на модификацию
# с единоразовым подсчетом якобиана.
EPS = 0.0001


# Функций скопированные из задания о нахождении ранга матрицы системы.
# ----------------------------------------------------------------------------------------------------------------------
def get_PAQ_LU(matrix):
    n = 0
    C = deepcopy(matrix)
    P = list(np.eye(SIZE))
    Q = list(np.eye(SIZE))

    for i in range(SIZE):
        pivotValue = 0
        pivot1 = -1
        pivot2 = -1
        for row in range(SIZE)[i:SIZE]:
            for column in range(SIZE)[i:SIZE]:
                if np.fabs(C[row][column]) > pivotValue:
                    pivotValue = np.fabs(C[row][column])
                    pivot1 = row
                    pivot2 = column
        if pivotValue != 0:
            if pivot1 != i:
                n += 1
                swap_rows(P, pivot1, i)
                swap_rows(C, pivot1, i)
            if pivot2 != i:
                n += 1
                swap_columns(Q, pivot2, i)
                swap_columns(C, pivot2, i)
            for j in range(SIZE)[i + 1:SIZE]:
                C[j][i] /= C[i][i]
                for s in range(SIZE)[i + 1:SIZE]:
                    C[j][s] -= C[j][i] * C[i][s]

    return C, P, Q, n


def swap_columns(your_list, pos1, pos2):
    for item in your_list:
        item[pos1], item[pos2] = item[pos2], item[pos1]


def swap_rows(input_array, row1, row2):
    temp = np.copy(input_array[row1][:])
    input_array[row1][:] = input_array[row2][:]
    input_array[row2][:] = temp


def trapez_rank(matrix):
    rank = SIZE
    for row in range(SIZE):
        if row_zero(matrix[SIZE - 1 - row]):
            rank -= 1
        else:
            break
    return rank


def row_zero(row):
    for i in range(SIZE):
        if abs(row[i]) - 0.0000001 > 0:
            return False
    return True


def eq_LUx_b(L, U, P, Q, b):
    op_num = 0
    b = list(np.array(P).dot(np.array(b).transpose()))
    y = []
    for i in range(SIZE):
        k = b[i]
        for j in range(SIZE)[0:i]:
            k -= L[i][j] * y[j]
            op_num += 2
        y.append(k)
    x = []
    for i in range(SIZE):
        k = y[SIZE - i - 1]
        for j in range(SIZE)[0:i]:
            k -= U[SIZE - i - 1][SIZE - j - 1] * x[j]
            op_num += 2
        x.append(k / U[SIZE - i - 1][SIZE - i - 1])
        op_num += 1
    x.reverse()
    x = np.array(Q).dot(np.array(x))
    op_num += 2 * SIZE * SIZE
    return x, op_num


def solve_degen(L, U, P, Q, b, rank):
    op_num = 0
    cU = np.copy(U)
    g = np.linalg.inv(L).dot(P).dot(b)
    op_num = 4 * SIZE * SIZE
    y = list(np.zeros(SIZE))
    for i in range(rank)[::-1]:
        g[i] = g[i] / cU[i, i]
        cU[i, :] /= cU[i, i]
        op_num += SIZE + 1
        for j in range(i):
            g[j] -= g[i] * cU[j, i]
            op_num += 2
            cU[j, :] -= cU[i, :] * cU[j, i]
            op_num += 2 + 2 * SIZE
        y[i] = g[i]
    x = np.array(Q).dot(np.array(y))
    op_num += SIZE * SIZE
    return x, op_num


# ----------------------------------------------------------------------------------------------------------------------
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
    for i in range(10):
        for j in range(10):
            res[i, j] = float(diff(Fs[i], xs[j]).subs(
                [(x1, a[0]), (x2, a[1]), (x3, a[2]), (x4, a[3]), (x5, a[4]), (x6, a[5]), (x7, a[6]), (x8, a[7]),
                 (x9, a[8]), (x10, a[9])]))
    return res


def slae_solve(matrix, _b):
    C, P, Q, _ = get_PAQ_LU(list(matrix))
    L = np.tril(np.array(C), -1) + np.identity(SIZE)
    U = np.triu(np.array(C), 0)
    rank = trapez_rank(U)
    Ab = np.zeros((SIZE, SIZE + 1))
    x = []
    n = 0
    for i in range(SIZE):
        Ab[:, i] = np.copy(matrix[:, i])
    Ab[:, SIZE] = np.copy(_b)
    if np.linalg.matrix_rank(Ab) == rank:
        if rank == SIZE:
            x, n = eq_LUx_b(L, U, P, Q, _b)
        else:
            x, n = solve_degen(L, U, P, Q, _b, rank)
    return x, n


def classic_newton_method(f, j, x0):
    x_cur = x0
    iter_num = 0
    operation_num = 0
    t = time.time()
    norm = 1000
    while norm > EPS:
        iter_num += 1
        apr_x, ops = slae_solve(j(x_cur), -f(x_cur))  # apr_x = x(k+1)-x(k)
        operation_num += ops
        norm = np.linalg.norm(apr_x, np.inf)
        x_cur = apr_x + x_cur
    duration = time.time() - t
    return x_cur, iter_num, operation_num, duration


def modified_newton_method(f, j, x0):
    x_cur = x0
    iter_num = 0
    operation_num = 0
    t = time.time()
    norm = 1000
    matrix = j(x_cur)
    C, P, Q, _ = get_PAQ_LU(list(matrix))
    L = np.tril(np.array(C), -1) + np.identity(SIZE)
    U = np.triu(np.array(C), 0)
    while norm > EPS:
        _b = -f(x_cur)
        rank = trapez_rank(U)
        Ab = np.zeros((SIZE, SIZE + 1))
        apr_x = []
        n = 0
        for i in range(SIZE):
            Ab[:, i] = np.copy(matrix[:, i])
        Ab[:, SIZE] = np.copy(_b)
        if np.linalg.matrix_rank(Ab) == rank:
            if rank == SIZE:
                apr_x, n = eq_LUx_b(L, U, P, Q, _b)
            else:
                apr_x, n = solve_degen(L, U, P, Q, _b, rank)
        operation_num += n
        iter_num += 1
        norm = np.linalg.norm(apr_x, np.inf)
        x_cur = apr_x + x_cur
    duration = time.time() - t
    return x_cur, iter_num, operation_num, duration


def newton_method_with_jump(f, j, x0, k):
    x_cur = x0
    iter_num = 0
    operation_num = 0
    t = time.time()
    for _ in range(k):
        iter_num += 1
        apr_x, ops = slae_solve(j(x_cur), -f(x_cur))  # apr_x = x(k+1)-x(k)
        operation_num += ops
        x_cur = apr_x + x_cur
    x_cur, itn, opn, _ = modified_newton_method(f, j, x_cur)
    duration = time.time() - t
    return x_cur, iter_num + itn, operation_num + opn, duration


def newton_method_with_period(f, j, x0, p):
    x_cur = x0
    iter_num = 0
    operation_num = 0
    t = time.time()
    norm = 1000
    matrix = j(x_cur)
    C, P, Q, _ = get_PAQ_LU(list(matrix))
    L = np.tril(np.array(C), -1) + np.identity(SIZE)
    U = np.triu(np.array(C), 0)
    period = 0
    while norm > EPS:
        period += 1
        if period == p:
            matrix = j(x_cur)
            C, P, Q, _ = get_PAQ_LU(list(matrix))
            L = np.tril(np.array(C), -1) + np.identity(SIZE)
            U = np.triu(np.array(C), 0)
            period = 0
        iter_num += 1
        _b = -f(x_cur)
        rank = trapez_rank(U)
        Ab = np.zeros((SIZE, SIZE + 1))
        apr_x = []
        n = 0
        for i in range(SIZE):
            Ab[:, i] = np.copy(matrix[:, i])
        Ab[:, SIZE] = np.copy(_b)
        if np.linalg.matrix_rank(Ab) == rank:
            if rank == SIZE:
                apr_x, n = eq_LUx_b(L, U, P, Q, _b)
            else:
                apr_x, n = solve_degen(L, U, P, Q, _b, rank)
        operation_num += n
        norm = np.linalg.norm(apr_x, np.inf)
        x_cur = apr_x + x_cur
    duration = time.time() - t
    return x_cur, iter_num, operation_num, duration


print("Результаты методов представленны в виде:(решение, количество итераций,"
      "количество арифметических операций, время выполнения)")
x_0 = np.array([0.5, 0.5, 1.5, -1, -0.5, 1.5, 0.5, -0.5, 1.5, -1.5])
print("x0 = " + str(x_0))
print("Метод Ньютона:")
x, i, u, d = classic_newton_method(func, jacobian, x_0)
print([x, i, u, d])
print("check:")
print(func(x))
print("Модифицированный метод Ньютона:")
x, i, u, d = modified_newton_method(func, jacobian, x_0)
print([x, i, u, d])
print("check:")
print(func(x))
k = 2
print("Метод Ньютона переходящий на модифицированный метод Ньютона после " + str(k) + "-й итерации:")
x, i, u, d = newton_method_with_jump(func, jacobian, x_0, k)
print([x, i, u, d])
print("check:")
print(func(x))
p = 5
print("Метод Ньютона с подсчетом якобиана каждые " + str(p) + " итераций:")
x, i, u, d = newton_method_with_period(func, jacobian, x_0, p)
print([x, i, u, d])
print("check:")
print(func(x))

x_0 = np.array([0.5, 0.5, 1.5, -1, -0.2, 1.5, 0.5, -0.5, 1.5, -1.5])
print("x0 = " + str(x_0))
print("Метод Ньютона:")
x, i, u, d = classic_newton_method(func, jacobian, x_0)
print([x, i, u, d])
print("check:")
print(func(x))
# Модифицированный метод расходится
# print("Модифицированный метод Ньютона:")
# print(modified_newton_method(func, jacobian, x_0))
k = 6
# При k<6 метод расходится
# При k>=6 метод сходится
# Вместо 7 стоит 6 видимо т.к. у меня переход происходит после к-й итерации, а видимо подразумевался переход с к-й.
print("Метод Ньютона переходящий на модифицированный метод Ньютона после " + str(k) + "-й итерации:")
x, i, u, d = newton_method_with_jump(func, jacobian, x_0, k)
print([x, i, u, d])
print("check:")
print(func(x))
p = 5
print("Метод Ньютона с подсчетом якобиана каждые " + str(p) + " итераций:")
x, i, u, d = newton_method_with_period(func, jacobian, x_0, p)
print([x, i, u, d])
print("check:")
print(func(x))
