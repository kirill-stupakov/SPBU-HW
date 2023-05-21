import matplotlib.pyplot as plt
import numpy as np
import sympy as sym
from matplotlib import cm

x = sym.symbols("x")
t = sym.symbols("t")


def grid(segment, n):
    a, b = segment
    step = (b - a) / n
    return np.array([a + i * step for i in range(n + 1)]), step


class TestConfig:
    def __init__(self, u, x_segment, N, time_segment, M, kappa):
        self.u = u
        self.N = N
        self.M = M
        self.x_segment = x_segment
        (self.x_grid, self.h) = grid(x_segment, N)
        self.time_segment = time_segment
        (self.time_grid, self.tau) = grid(time_segment, M)
        self.kappa = kappa
        self.f = sym.diff(u, t, 1) - kappa * sym.diff(u, x, 2)
        self.mu = u.subs(t, 0)
        (a, b) = x_segment
        self.mu_1 = u.subs(x, a)
        self.mu_2 = u.subs(x, b)


def explicit_schema(config: TestConfig):
    # (u_n_next - u_n) / tau = k/h^2 (u_n-1 - 2u_n + u_n+1) + f(x_n,t)
    U = np.zeros((config.N + 1, config.M + 1), dtype=float)

    for i in range(0, config.N + 1):
        U[i, 0] = config.mu.subs(x, config.x_grid[i])
    for j in range(0, config.M + 1):
        U[0, j] = config.mu_1.subs(t, config.time_grid[j])
        U[config.N, j] = config.mu_2.subs(t, config.time_grid[j])

    for k in range(1, config.M + 1):
        for i in range(1, config.N):
            U[i, k] = (
                config.tau
                * config.kappa
                * (U[i - 1, k - 1] - 2 * U[i, k - 1] + U[i + 1, k - 1])
                / (config.h**2)
                + config.tau
                * config.f.subs([(x, config.x_grid[i]), (t, config.time_grid[k - 1])])
                + U[i, k - 1]
            )
    x_grid, y_grid = np.meshgrid(config.x_grid, config.time_grid, indexing="ij")
    return x_grid, y_grid, U


def implicit_schema(config: TestConfig):
    U = np.zeros((config.N + 1, config.M + 1), dtype=float)
    A, C = [np.zeros(config.N + 1, dtype=float) for _ in range(2)]
    B = np.ones(config.N + 1, dtype=float)
    D = np.zeros(config.N + 1, dtype=sym.Symbol)
    for k in range(1, config.M + 1):
        D[0] = config.mu_1.subs(t, config.time_grid[k])
        D[config.N] = config.mu_2.subs(t, config.time_grid[k])
        for i in range(0, config.N + 1):
            U[i, 0] = config.mu.subs(x, config.x_grid[i])
        for i in range(1, config.N):
            A[i] = config.kappa / config.h**2
            B[i] = -2 * config.kappa / config.h**2 - 1 / config.tau
            C[i] = config.kappa / config.h**2
            D[i] = -U[i, k - 1] / config.tau - config.f.subs(
                [(x, config.x_grid[i]), (t, config.time_grid[k])]
            )
        s = np.zeros(config.N + 1, dtype=float)
        t1 = np.zeros(config.N + 1, dtype=float)
        s[0] = -C[0] / B[0]
        t1[0] = D[0] / B[0]
        for i in range(1, config.N + 1):
            s[i] = -C[i] / (A[i] * s[i - 1] + B[i])
            t1[i] = (D[i] - A[i] * t1[i - 1]) / (A[i] * s[i - 1] + B[i])
        U[config.N, k] = t1[config.N]
        for i in range(config.N - 1, -1, -1):
            U[i, k] = s[i] * U[i + 1, k] + t1[i]
    x_grid, t_grid = np.meshgrid(config.x_grid, config.time_grid, indexing="ij")
    return x_grid, t_grid, U


def get_test_case():
    cases = []

    kappas = [1e-3, 1e-2, 1e-1]
    for kappa in kappas:
        cases.append(TestConfig(x**2 / 4 + t**2 / 4, (0, 10), 5, (0, 10), 5, kappa))
    for kappa in kappas:
        cases.append(
            TestConfig(x**2 / 4 + t**2 / 4, (0, 10), 100, (0, 10), 100, kappa)
        )
    for kappa in kappas:
        cases.append(TestConfig(t * x, (0, 10), 100, (0, 10), 100, kappa))
    for kappa in kappas:
        cases.append(
            TestConfig(
                x * t**3 - 2 * x + 25 - x**5, (0, 10), 100, (0, 10), 100, kappa
            )
        )

    index = int(input(f"Enter test case number (1 - {len(cases)}): "))
    return cases[index - 1]


def draw(plts_data, title):
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122, projection="3d")

    def set_next_fig(ax, data):
        X_grid, Y_grid, U = data
        ax.plot_surface(X_grid, Y_grid, U, cmap=cm.coolwarm)
        ax.set_title(title)

    set_next_fig(ax1, plts_data[0])
    set_next_fig(ax2, plts_data[1])

    plt.show()


def main():
    test_case = get_test_case()
    explicit = explicit_schema(test_case)
    implicit = implicit_schema(test_case)
    draw([explicit, implicit], f"kappa = {test_case.kappa}")


if __name__ == "__main__":
    main()
