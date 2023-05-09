import time
from numpy.linalg import inv
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import random, eye

class LinearEquationSolver:
    left_side: np.ndarray
    right_side: np.ndarray
    precision: float

    def set_left_side(self, ls: np.ndarray):
        self.left_side = ls

    def set_right_side(self, rs: np.ndarray):
        self.right_side = rs

    def set_precision(self, precision: float):
        self.precision = precision

    def solve_simple_iteration(self):
        b = np.zeros_like(self.left_side)
        c = np.zeros_like(self.right_side)

        for i in range(b.shape[0]):
            for j in range(b.shape[1]):
                if i != j:
                    b[i][j] = -self.left_side[i][j] / self.left_side[i][i]

        for i in range(c.size):
            c[i] = self.right_side[i] / self.left_side[i][i]

        prev_solution = None
        current_solution = np.zeros_like(c)
        norm = np.linalg.norm(b)
        iterations = 0

        while prev_solution is None or (norm * np.linalg.norm(current_solution - prev_solution)) / (1 - norm) > self.precision:
            iterations += 1
            prev_solution = current_solution
            current_solution = b @ prev_solution + c

        return current_solution, iterations

    def solve_zeidel(self):
        L = np.zeros_like(self.left_side)
        D = np.zeros_like(self.left_side)
        R = np.zeros_like(self.left_side)

        for i in range(self.left_side.shape[0]):
            for j in range(self.left_side.shape[1]):
                elem = self.left_side[i][j]
                if i < j:
                    R[i][j] = elem
                elif i > j:
                    L[i][j] = elem
                else:
                    D[i][j] = elem

        dinv = inv(D + L)
        b = -dinv @ R
        c = dinv @ self.right_side

        prev_solution = None
        current_solution = np.zeros_like(c)
        norm = np.linalg.norm(b)
        iterations = 0

        while prev_solution is None or (norm * np.linalg.norm(current_solution - prev_solution)) / (1 - norm) > self.precision:
            iterations += 1
            prev_solution = current_solution
            current_solution = b @ prev_solution + c

        return current_solution, iterations

def get_test_case():
    cases = [
        [
            np.array([
                [2, 0.5, 1],
                [0.3, 3, 0.3],
                [1, 1, 2]
            ]),
            np.array([10, 11, 12]),
            1e-3,
            1e-6
        ],
        [
            (random(256, 256, 0.05) + eye(256) * 128).toarray(),
            np.random.rand(256),
            1e-3,
            1e-16
        ],
        [
            (random(1024, 1024, 0.15) + eye(1024) * 256).toarray(),
            np.random.rand(1024),
            1e-3,
            1e-9
        ]
    ]

    index = int(input(f"Enter test case number (1 - {len(cases)}): "))
    return cases[index - 1]

def main():
    [left_side, right_side, precision_start, precision_end] = get_test_case()

    solver = LinearEquationSolver()
    solver.set_left_side(left_side)
    solver.set_right_side(right_side)

    current_precision = precision_start
    precisions = []
    iterations_simple = []
    iterations_zeidel = []

    while current_precision > precision_end:
        solver.set_precision(current_precision)
        _, simple_iterations = solver.solve_simple_iteration()
        _, zeidel_iterations = solver.solve_zeidel()

        precisions.append(current_precision)
        iterations_simple.append(simple_iterations)
        iterations_zeidel.append(zeidel_iterations)
        current_precision /= 2

    plt.plot(precisions, iterations_simple)
    plt.plot(precisions, iterations_zeidel)
    plt.xscale("log")
    plt.show()

if __name__ == "__main__":
    main()

