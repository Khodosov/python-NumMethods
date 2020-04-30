from copy import deepcopy
import numpy as np

SIZE = 3
A_arr = [[0.0, 7.0, 9.0],
         [3.0, 8.0, 5.0],
         [5.0, 7.0, 3.0]]

A_arr = list(np.random.uniform(0, 10, (SIZE, SIZE)))


def print_matrix(matrix):
    print("----------------------------------------------------------------------------------")
    for arr in matrix:
        print(arr, "\n")
    print("----------------------------------------------------------------------------------")


# def get_LU(matrix):
#     U = deepcopy(matrix)
#     L = [[0] * SIZE for _ in range(SIZE)]
#     for i in range(SIZE):
#         L[i][i] = 1
#
#     for i in range(SIZE - 1):
#         for j in range(SIZE)[i + 1:SIZE]:
#             k = round(U[j][i] / U[i][i], 3)
#             L[j][i] = k
#             for s in range(SIZE):
#                 U[j][s] = round(U[j][s] - k * U[i][s], 3)
#
#     return U, L


def get_PA_LU(matrix):
    n = 0
    C = deepcopy(matrix)
    P = [[0] * SIZE for _ in range(SIZE)]
    for i in range(SIZE):
        P[i][i] = 1

    for i in range(SIZE):
        pivotValue = 0
        pivot = -1
        for row in range(SIZE)[i:SIZE]:
            if np.fabs(C[row][i]) > pivotValue:
                pivotValue = np.fabs(C[row][i])
                pivot = row
        if pivotValue != 0:
            n += 1
            swap_rows(P, pivot, i)
            swap_rows(C, pivot, i)
            for j in range(SIZE)[i + 1:SIZE]:
                C[j][i] /= C[i][i]
                for s in range(SIZE)[i + 1:SIZE]:
                    C[j][s] -= C[j][i] * C[i][s]

    return C, P, n


def swap_columns(input_array, col1, col2):
    temp = np.copy(input_array[:][col1])
    input_array[:][col1] = input_array[:][col2]
    input_array[:][col2] = temp


def swap_rows(input_array, row1, row2):
    temp = np.copy(input_array[row1][:])
    input_array[row1][:] = input_array[row2][:]
    input_array[row2][:] = temp


# U_arr, L_arr = get_LU(A_arr)
# print("U")
# print_matrix(U_arr)
# print("L")
# print_matrix(L_arr)
# print("A")
# print_matrix(A_arr)

print("A:")
print(np.array(A_arr))

print("PA=LU")
C, P, n = get_PA_LU(A_arr)

# C = L + U - E
print("C")
print(np.array(C))

print("P")
print(np.array(P))

print("L")
L = np.tril(np.array(C), -1) + np.identity(SIZE)
print(L)

print("U")
U = np.triu(np.array(C), 0)
print(U)

print("L*U")
print(L.dot(U))

print("P*A")
PA = np.array(P).dot(np.array(A_arr))
print(PA)


# a) determinant
# |A|=|invP|*|L|*|U|=(-1)^n * |U|
def det_LU(U, n):
    out = (-1) ** n
    for i in range(SIZE):
        out *= U[i][i]
    return out


print("determinant A:")
print(det_LU(U, n))


# b) A*x = b => P*A*x = L*U*x = P*b
def eq_LUx_b(L, U, P, b):
    b = list(np.array(P).dot(np.array(b).transpose()))
    y = []
    for i in range(SIZE):
        k = b[i]
        for j in range(SIZE)[0:i]:
            k -= L[i][j] * y[j]
        y.append(k)
    x = []
    for i in range(SIZE):
        k = y[SIZE - i - 1]
        for j in range(SIZE)[0:i]:
            k -= U[SIZE - i - 1][SIZE - j - 1] * x[j]
        x.append(k / U[SIZE - i - 1][SIZE - i - 1])
    x.reverse()
    return x


print("A*x = L*U*x = b")
print("b")
b = list(np.random.uniform(0, 10, SIZE))
print(b)
print("x")
x = eq_LUx_b(L, U, P, b)
print(x)
print("check")
print(np.array(A_arr).dot(np.transpose(np.array(x))))


# c) inverse matrix
def inverse(L, U, P):
    invA = []
    eye = np.eye(SIZE)
    for i in range(SIZE):
        invA.append(eq_LUx_b(L, U, P, list(eye)[i]))
    return np.transpose(invA)


print("Inversed A:")
invA = inverse(L, U, P)
print(np.array(invA))
print("invA*A")
print(np.array(invA).dot(np.array(A_arr)))
print("A*invA")
print(np.array(A_arr).dot(np.array(invA)))

# d) Число обусловленности A
print("mu")
print(np.linalg.norm(np.array(invA)) * np.linalg.norm(np.array(A_arr)))