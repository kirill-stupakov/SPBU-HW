from enum import Enum, auto

import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg
from numpy.linalg import norm

ITERATION_LIMIT = 500


def dim(matrix):
    return matrix.shape[0]


def hilbert(n, precision=16):
    return np.fromfunction(lambda i, j: np.round(1 / (i + j + 1), precision), (n, n))


class DrawProperty(Enum):
    ONLY_AMOUNT = auto()
    ONLY_ERROR = auto()


def get_test_case():
    cases = [
        [
            np.array([[-198.1, 389.9, 123.2], [0, 202.4, 249.3], [0, 0, 489.2]]),
            np.array([[-0.862254], [0.0249838], [-2.32304]]),
            "Upper triangular matrix",
        ],
        [
            np.array(
                [
                    [2, -1, 0, 0, 0],
                    [-3, 8, -1, 0, 0],
                    [0, -5, 12, 2, 0],
                    [0, 0, -6, 18, -4],
                    [0, 0, 0, -5, 10],
                ]
            ),
            np.array([[0.92884], [0.630153], [0.580092], [-0.200029], [-0.200029]]),
            "Threediagonal matrix",
        ],
        [
            np.array([[1, 0.99], [0.99, 0.98]]),
            np.array([[0.715274], [-1.17753]]),
            "2x2 example matrix",
        ],
        [
            np.array([[-198.1, 389.9, 123.2], [0, 202.4, 249.3], [0, 0, 489.2]]),
            np.array([[-0.862254], [0.0249838], [-2.32304]]),
            "Upper triangular matrix",
        ],
        [
            np.array([[-401.98, 200.34], [1202.04, -602.32]]),
            np.array([[0.715274], [-1.17753]]),
            "2x2 example matrix",
        ],
        [
            hilbert(4),
            np.array([[1.4514], [-1.99799], [1.83011], [-0.471568]]),
            "4th order Hilbert matrix",
        ],
        [
            hilbert(5),
            np.array([[-0.943337], [-1.25822], [-1.39549], [0.738115], [-0.0660465]]),
            "5th order Hilbert matrix",
        ],
        [
            hilbert(6),
            np.array(
                [[1.07713], [0.232377], [2.22464], [-1.12486], [1.80719], [-0.113347]]
            ),
            "6th order Hilbert matrix",
        ],
        [hilbert(10), np.random.rand(10, 1), "10th order Hilbert matrix"],
        [hilbert(16), np.random.rand(16, 1), "16th order Hilbert matrix"],
    ]

    index = int(input(f"Enter test case number (1 - {len(cases)}): "))
    return cases[index - 1]


def calculate_power(matrix, x, eps):
    x_current = matrix @ x
    x_previous = x
    iterations = 1
    value_previous = x_current[0][0] / x_previous[0][0]
    x_previous = x_current
    x_current = matrix @ x_current
    value_current = x_current[0][0] / x_previous[0][0]
    while (iterations < ITERATION_LIMIT) and (
        norm(value_current - value_previous) > eps
    ):
        value_previous = value_current
        x_previous = x_current
        x_current = matrix @ x_current
        value_current = x_current[0][0] / x_previous[0][0]
        iterations += 1

    return np.abs(value_current), iterations


def calculate_scalar(matrix, x, eps):
    x_current = matrix @ x
    x_previous = x
    y_current = matrix.T @ x_previous
    value_previous = (x_current.T @ y_current)[0][0] / (x_previous.T @ x)[0][0]
    iterations = 1

    x_previous = x_current
    x_current = matrix @ x_current
    y_current = matrix.T @ y_current

    value_current = (x_current.T @ y_current)[0][0] / (x_previous.T @ y_current)[0][0]
    while (iterations < ITERATION_LIMIT) and (
        np.abs(value_current - value_previous) > eps
    ):
        value_previous = value_current
        x_previous = x_current
        x_current = matrix @ x_current
        y_current = matrix.T @ y_current

        value_current = (x_current.T @ y_current)[0][0] / (x_previous.T @ y_current)[0][
            0
        ]
        iterations += 1

    return np.abs(value_current), iterations


def process_test(test_element):
    [matrix, x, title] = test_element

    errors_power = []
    errors_scalar = []
    iterations_power = []
    iterations_scalar = []

    eigen_values = linalg.eigvals(matrix)
    eigen_value = eigen_values[0]
    for i in range(len(eigen_values)):
        if norm(abs(eigen_values[i]) > abs(eigen_value)):
            eigen_value = eigen_values[i]

    eigen_value = eigen_value.real

    from_eps_degree = -5
    to_eps_degree = -2
    epsilons = np.logspace(from_eps_degree, to_eps_degree, 300)
    for eps in epsilons:
        new_eigen_value_power, current_iterations_power = calculate_power(
            matrix, x, eps
        )
        new_eigen_value_scalar, current_iterations_scalar = calculate_scalar(
            matrix, x, eps
        )
        errors_power.append(norm(np.abs(eigen_value) - np.abs(new_eigen_value_power)))
        errors_scalar.append(norm(np.abs(eigen_value) - np.abs(new_eigen_value_scalar)))
        iterations_power.append(current_iterations_power)
        iterations_scalar.append(current_iterations_scalar)

    _, axis = plt.subplots(1, 2, figsize=(16, 8))

    axis[0].set_xscale("log")
    axis[0].set_title(title)
    axis[0].set_xlabel("Target precision")
    axis[0].set_ylabel("Error")
    axis[0].plot(epsilons, errors_power, label="Power method")
    axis[0].plot(epsilons, errors_scalar, label="Scalar products method")

    axis[1].set_xscale("log")
    axis[1].set_title(title)
    axis[1].set_xlabel("Target precision")
    axis[1].set_ylabel("Iterations")
    axis[1].plot(epsilons, iterations_power, label="Power method")
    axis[1].plot(epsilons, iterations_scalar, label="Scalar products method")

    plt.legend()
    plt.show()


def main():
    test_case = get_test_case()

    print(test_case[2])
    process_test(test_case)


if __name__ == "__main__":
    main()
