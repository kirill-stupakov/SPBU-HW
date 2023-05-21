from math import sqrt
import numpy as np
from numpy.linalg import norm, inv, det, solve
from scipy.sparse import random, eye


# spectral conditionality criterion
def cond_s(A):
    return round(norm(A) * norm(inv(A)), 2)


# volumetric criterion
def cond_v(A):
    numerator = 1
    for n in range(0, dim(A)):
        row_sum = 0
        for m in range(0, dim(A)):
            row_sum += A.item(n, m) ** 2
        numerator *= sqrt(row_sum)
    return round(numerator / abs(det(A)), 2)


# angular criterion
def cond_a(A):
    C = inv(A)
    cond = 0
    for i in range(0, dim(A)):
        cond = max(cond, norm(A[i, :]) * norm([C[:, i]]))
    return round(cond, 2)


def vander(n):
    def element(i, j):
        a = lambda k: k / 2
        b = lambda k: k + 1
        return a(i + 1) ** b(j + 1)

    return np.fromfunction(lambda i, j: element(i, j), (n, n))

def hilbert(n, precision=16):
    return np.fromfunction(lambda i, j: np.round(1 / (i + j + 1), precision), (n, n))

def sparse(n):
    return (random(n, n, 0.05) + eye(n) * (n / 2)).toarray()

def dim(A):
    return A.shape[0]


def rand_x(n, max_value=1000):
    return np.random.randint(-max_value, max_value, n)


def lu_decomposition(A, b):
    n = dim(A)
    a_i = A
    b_i = b
    L = np.eye(n)
    for j in range(0, n - 1):
        m_i= np.eye(n)
        for i in range(j + 1, n):
            m_i[i][j] = -a_i[i][j] / a_i[j][j]
        L = L @ (-m_i + 2 * np.eye(n))
        a_i = m_i @ a_i
        b_i = m_i @ b_i
    U = a_i
    return L, U, b_i


def solve_with_LU(A, b):
    L, U, b_1 = lu_decomposition(A, b)
    x = b

    for i in range(dim(A)):
        x[i] /= L[i][i]
        L[i][i] = 1
        for k in range(i + 1, dim(A)):
            x[k] -= L[k][i] * x[i]
            L[k][i] = 0

    for i in range(dim(A) - 1, -1, -1):
        x[i] /= U[i][i]
        U[i][i] = 1
        for k in range(0, i):
            x[k] -= U[k][i] * x[i]
            U[k][i] = 0
    return x, L, U, b_1


def print_conds(matrices):
    for [A, matrix_name] in matrices:
        print(
            f"s({matrix_name}) = {cond_s(A):.2e}, v({matrix_name}) = {cond_v(A):.2e}, a({matrix_name}) = {cond_a(A):.2e}")


def example_with_matrix(A, x):
    b = A @ x
    solution_with_lu, L, U, _ = solve_with_LU(A, b)
    diff_with_lu = norm(solution_with_lu - x)
    diff_without_lu = norm(solve(A, b) - x)

    print_conds([[A, "A"], [L, "L"], [U, "U"]])
    print(f"Difference with LU = {diff_with_lu}")
    print(f"Difference without LU = {diff_without_lu}")
    print(f"Difference between methods = {abs(diff_without_lu - diff_with_lu)}")


def main():
    n = 8
    print(f"Vander matrix ({n})")
    example_with_matrix(vander(n), np.ones((n, 1)))
    print()

    print(f"Hilbert matrix ({n})")
    example_with_matrix(hilbert(n), np.ones((n, 1)))
    print()

    print(f"Sparse matrix ({n})")
    example_with_matrix(sparse(n), np.ones((n, 1)))
    print()

if __name__ == '__main__':
    main() 

