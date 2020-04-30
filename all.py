import numpy as np


class LUdecomposition:
    LU_matrix = np.array([])
    P = []
    Q = []
    rank = 0
    det = 0


def getMaximum(Ma, k1, k2):
    maximum = [0, 0, 0]
    for row in range(k1, k2):
        for col in range(k1, k2):
            now = abs(Ma[row][col])
            if now > maximum[0]:
                maximum = [now, row, col]
    return maximum

def LU(A):
    n = len(A)
    LUmatrix = A.copy()
    rank = 0
    P = list(range(n))
    Q = list(range(n))
    perms = 0
    for k in range(n):  # (k + 1)-ый шаг ПХ Гаусса
        #  Нахождение максимального элемента в оставшейся матрице
        maxEl, maxRow, maxCol = getMaximum(LUmatrix, k, n)
        if abs(maxEl) < 10 ** (-10):
            break  # Матрица приведена к трапецевидной форме
        else:
            rank += 1
        # Перестановка столбцов и строк
        if maxCol != k:
            perms += 1
            LUmatrix[:, [k, maxCol]] = LUmatrix[:, [maxCol, k]]  # Пересатновка столбцов
            Q[k], Q[maxCol] = Q[maxCol], Q[k]
        if maxRow != k:  # Перестановка строк
            perms += 1
            LUmatrix[[k, maxRow], :] = LUmatrix[[maxRow, k], :]
            P[k], P[maxRow] = P[maxRow], P[k]
        for j in range(k + 1, n):  # Обнуление k-ого столбца: вычитание из j-ой строчки k-ую * l(j, k)
            l = LUmatrix[j][k] / LUmatrix[k][k]
            LUmatrix[j, k:] = LUmatrix[j, k:] - l * LUmatrix[k, k:]
            LUmatrix[j][k] = l
    det = pow(-1, perms) * np.prod(LUmatrix.diagonal())

    result = LUdecomposition()
    result.LU_matrix, result.P, result.Q, result.rank, result.det = LUmatrix, P, Q, rank, det
    print('LU:', LUmatrix, 'P:', P, 'Q:', Q, 'rank:', rank, 'Python rank:', np.linalg.matrix_rank(A), sep='\n')
    print('det:', det, 'Python det:', np.linalg.det(A), sep='\n')
    if rank != np.linalg.matrix_rank(A):
        print('A =\n', A)
    return result


def checkLUequalPAQ(L, U, P, A1, Q):
    A = A1.copy()
    for i in range(len(A)):
        A[i] = A1[P[i]]
    A2 = A.copy()
    for j in range(len(A)):
        A[:, j] = A2[:, Q[j]]
    print('PAQ =', '\n', A)
    print('LU = ', '\n', np.dot(L, U))


def getEquationAx_b(Lu, b):
    M = Lu.LU_matrix
    N = len(M)
    Pb = b.copy()
    for i in range(N):
        Pb[i] = b[Lu.P[i]]
    y = np.zeros((N, 1))
    # Ly = Pb
    y[0] = Pb[0]
    for i in range(1, N):
        y[i] = Pb[i] - np.dot(M[i, :i], y[:i])
    z = np.zeros((N, 1))
    # Uz = y
    z[-1] = y[-1] / M[-1][-1]
    for i in range(N - 2, -1, -1):
        z[i] = (y[i] - np.dot(M[i, i + 1:], z[i + 1:])) / M[i][i]
    # x = Qz
    x = z.copy()
    for i in range(N):
        x[Lu.Q[i]] = z[i]
    return x


def getinverseMatrix(Lu):
    N = len(Lu.LU_matrix)
    X = np.zeros((N, 1))
    for i in range(N):
        e = np.zeros((N, 1))
        e[i] = 1
        x = getEquationAx_b(Lu, e)
        X = np.column_stack((X, x))
    return X[:, 1:]


def getMaxRowTotal(A, Ainv):  # Максимум сумм строк (p = inf)
    return max([sum(map(abs, a)) for a in A]) * max([sum(map(abs, a)) for a in Ainv])


def isAcompatible(Lu, b):
    N = len(L)
    M = Lu.LU_matrix
    Pb = b.copy()
    for i in range(N):
        Pb[i] = b[Lu.P[i]]
    y = np.zeros((N, 1))
    y[0] = Pb[0]
    for i in range(1, N):
        y[i] = Pb[i] - np.dot(M[i, :i], y[:i])
    zerosInTheEnd = 0
    for i in range(Lu.rank, N):
        if y[i] < 10 ** (-8):
            zerosInTheEnd += 1
    if zerosInTheEnd == N - Lu.rank:
        print("Система совместна")
        z = np.zeros((N, 1))
        for i in range(Lu.rank - 1, -1, -1):
            z[i] = (y[i] - np.dot(M[i, i + 1:], z[i + 1:])) / M[i][i]
        x = z.copy()
        for i in range(N):
            x[Lu.Q[i]] = z[i]
        print("Частное решение:\n", x)
    else:
        print("Система не является совместной")


def QR(A1):
    R = A1.copy()
    n = len(R)
    Q = []
    for i in range(0, n - 1):
        cur_col = R[i:, [i]]
        alpha = np.sign(-cur_col[0]) * np.linalg.norm(cur_col)
        e = np.zeros((n - i, 1))
        e[0] = 1
        w = cur_col - alpha * e
        w /= np.linalg.norm(w)
        U = np.eye(n)
        U[i:, i:] = np.eye(n - i) - 2 * np.dot(w, w.transpose())
        print(U)
        R = np.dot(U, R)
        Q = U.copy() if i == 0 else np.dot(Q, U)
    return Q, R


def solutionRx_Qb(Q, R, b):
    N = len(R)
    Qb = np.dot(Q.transpose(), b)
    x = np.zeros((N, 1))
    x[-1] = Qb[-1] / R[-1][-1]
    for i in range(N - 2, -1, -1):
        x[i] = (Qb[i] - np.dot(R[i, i + 1:], x[i + 1:])) / R[i][i]
    print("Решение системы Rx=Qb\n", x)
    print('Python:\n', np.linalg.solve(np.linalg.qr(A)[1], np.dot(np.linalg.qr(A)[0].transpose(), b)))
    print('Точность\n', np.dot(A, x) - b)


N = 3
# Тесты
A = np.random.rand(N, N)
b = np.random.rand(N, 1)
print("Матрица А:\n", A, '\n', '-' * 50)
print("LU разложение")
Lu = LU(A)
print('-' * 50)
U = np.triu(Lu.LU_matrix)  # Берем L и U из одной матрицы
L = np.tril(Lu.LU_matrix)
for i in range(len(L)):
    L[i][i] = 1 if U[i][i] != 0 else 0
P = Lu.P
Q = Lu.Q
print("LU = PAQ")
checkLUequalPAQ(L, U, P, A, Q)
if abs(Lu.det) > 10 ** (-10):  # Если не ноль или не близко к нему
    print('-' * 50, '\nКастомное решение Ax = b: ')
    x = getEquationAx_b(Lu, b)
    print(x, "\n\nРешение Python:")
    print(np.linalg.solve(A, b))
    print("Ax - b = ")
    print(np.dot(A, x) - b, '\n', '-' * 50)
    print("Кастомная обратная матрица:")
    Ainv = getinverseMatrix(Lu)
    print(Ainv, '\n', '-' * 50, '\nPython:')
    print(np.linalg.inv(A), end='\n\n')  # - Обратная
    print("Ainv*A = E:\n", np.dot(Ainv, A), end='\n\n')
    print("A*Ainv = E:\n", np.dot(A, Ainv), '\n\n', '-' * 50)
    print('Число обусловленности inf:\n', getMaxRowTotal(A, Ainv), end='\n\n')
    print('Python:\n', np.linalg.cond(A, np.inf), '\n', '-' * 50)  # Число обусловленности
    print("QR разложение:\n")
    qr = QR(A)
    Q = qr[0]
    R = qr[1]
    print("Q =\n", Q)
    print("Python:\n", np.linalg.qr(A)[0], '\n')
    print("R =\n", R)
    print("Python:\n", np.linalg.qr(A)[1], '\n', '-' * 50)
    print("A == QR - ?\n", np.dot(Q, R), '\n')
    solutionRx_Qb(Q, R, b)
else:
    print("Матрица вырождена")
    isAcompatible(Lu, b)
    print("Python:\n", np.linalg.solve(A, b))