import math
import time
from collections.abc import Callable
import numpy as np
import matplotlib.pyplot as plt

RationalFunction = Callable[[float], float]

class GridMethodSolver:
    p: RationalFunction
    q: RationalFunction
    r: RationalFunction
    f: RationalFunction
    a: float
    alpha_1: float
    alpha_2: float
    alpha: float
    b: float
    beta_1: float
    beta_2: float
    beta: float
    target_precision: float


    def __make_grid(self, interval_count: int):
        h = (self.b - self.a) / interval_count
        return [self.a + i * h for i in range(interval_count + 1)]

    def __make_equation_system(self, grid: list[float]):
        left_side_compressed: list[list[float]] = []
        right_side: list[float] = []
        step = grid[1] - grid[0]

        left_side_compressed.append([0, self.alpha_1 + self.alpha_2 / step, -self.alpha_2 / step])
        right_side.append(self.alpha)

        for i in range(1, len(grid) - 1):
            current_row: list[float] = []
            current_row.append(-self.p(grid[i]) / (step ** 2) - self.q(grid[i]) / (2 * step))
            current_row.append(2 * self.p(grid[i]) / (step ** 2) + self.r(grid[i]))
            current_row.append(-self.p(grid[i]) / (step ** 2) + self.q(grid[i]) / (2 * step))
            left_side_compressed.append(current_row)
            right_side.append(self.f(grid[i]))

        left_side_compressed.append([-self.beta_2 / step, self.beta_1 + self.beta_2 / step, 0])
        right_side.append(self.beta)

        return left_side_compressed, right_side

    def __solve_threediagonal_matrix(self, left_side_compressed: list[list[float]], right_side: list[float]):
        s: list[float] = [-left_side_compressed[0][2] / left_side_compressed[0][1]]
        t: list[float] = [right_side[0] / left_side_compressed[0][1]]
        for i in range(1, len(right_side)):
            s.append(left_side_compressed[i][2] / (-left_side_compressed[i][1] - left_side_compressed[i][0] * s[i - 1]))
            t.append((left_side_compressed[i][0] * t[i - 1] - right_side[i]) / (-left_side_compressed[i][1] - left_side_compressed[i][0] * s[i - 1]))

        solution: list[float] = [0] * len(right_side)
        solution[-1] = t[-1]
        for i in range(len(solution) - 2, -1, -1):
            solution[i] = s[i] * solution[i + 1] + t[i]

        return solution

    def __get_solution_difference(self, sol1: list[float], sol2: list[float]):
        return np.max([abs(sol1[i] - sol2[i // 2]) for i in range(len(sol1))])

    def solve(self):
        print("Running...")
        start_time = time.perf_counter()

        intervals: list[int] = []
        precisions: list[float] = []
        current_interval_count = 8
        current_precision = self.target_precision + 1 
        prev_solution = None
        grid = None
        while current_precision > self.target_precision and current_interval_count < 1e7:
            grid = self.__make_grid(current_interval_count)
            (left_side, right_side) = self.__make_equation_system(grid)
            current_solution = self.__solve_threediagonal_matrix(left_side, right_side)
            if prev_solution:
                difference = self.__get_solution_difference(current_solution, prev_solution)
                intervals.append(current_interval_count)
                precisions.append(difference)
                current_precision = difference

            prev_solution = current_solution
            current_interval_count *= 2
            
        end_time = time.perf_counter()
        print(f"Done ({(end_time - start_time):0.4f}s)")

        _, axis = plt.subplots(2, 1)
        axis[0].plot(grid, prev_solution)
        axis[0].set_title("Solution")
        axis[1].plot(intervals, precisions)
        axis[1].set_title("Precision")
        plt.xscale("log")
        plt.yscale("log")
        plt.show()

    def set_functions(self, p: RationalFunction, q: RationalFunction, r: RationalFunction, f: RationalFunction):
        self.p = p
        self.q = q
        self.r = r
        self.f = f

    def set_boundary_conditions(self, a: float, alpha_1: float, alpha_2: float, alpha: float, b: float, beta_1: float,
                                beta_2: float, beta: float):
        self.a = a
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.alpha = alpha
        self.b = b
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.beta = beta

    def set_precision(self, epsilon: float):
        self.target_precision = epsilon

def get_test_case():
    cases = [
        [    
             [lambda x: x ** 2 + 1, lambda x: -x, lambda x: 1, lambda x: x],
             [-1, 1, 0, 0, 1, 1, 0, 0],
             1e-7,
        ],
        [    
             [lambda x: math.sin(x) + 2, lambda x: math.cos(x), lambda x: 3 * x + 1, lambda x: 0],
             [0, 1, 1, 2, 1, 1, 1, 1],
             1e-7,
        ],
        [    
             [lambda x: 1 / (x- 3), lambda x: 1 + x / 2, lambda x: math.exp(x / 2), lambda x: 2 - x],
             [-1, 1, 0, 0, 1, 1, 0, 1],
             1e-6,
        ],
        [    
             [lambda x: -1, lambda x: - math.cos(x) / (1 + x), lambda x: -2 + x, lambda x: x + 1],
             [0, 0.2, 1, -0.8, 1, 0.9, 1, -0.1],
             1e-7,
        ]
    ]

    index = int(input(f"Enter test case number (1 - {len(cases)}): "))
    return cases[index - 1]

def main():
    [functions, boundary_conditions, precision] = get_test_case()

    solver = GridMethodSolver()
    solver.set_functions(*functions)
    solver.set_boundary_conditions(*boundary_conditions)
    solver.set_precision(precision)
    solver.solve()


if __name__ == "__main__":
    main()
