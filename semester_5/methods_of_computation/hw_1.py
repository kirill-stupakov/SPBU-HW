import numpy as np

def function(x):
  return 1.2 * x ** 4 + 2 * x ** 3 - 13 * x ** 2 - 14.2 * x - 24.1

def first_derivative(x):
  return 4.8 * x ** 3 + 6 * x ** 2 - 26 * x - 14.2

def second_derivative(x):
  return 14.4 * x ** 2 + 12 * x - 26

def get_sign_change_intervals(function, a, b, n):
  intervals = []
  division_step = (b - a) / n

  y_0, y_1 = 0, function(a)
  for x in np.arange(a, b, division_step):
    y_0 = y_1
    y_1 = function(x + division_step)

    if (y_0 * y_1 <= 0):
      intervals.append([x, x + division_step])

  return intervals

def print_method_result(name, result):
  (initial_root, steps, root, length, abs_root) = result
  print(f'{name}: x_0 = {initial_root}, m = {steps}, x_m = {root}, |x_m - x_m-1| = {length}, |f(x_m) - 0| = {abs_root}')

def bisection_method(function, a, b, precision):
  step_count = 0
  current_precision = (b - a) / 2
  midpoint = (a + b) / 2
  initial_midpoint = midpoint

  while current_precision > precision:
    if function(a) == 0:
      return (initial_midpoint, step_count, a, b - a, abs(function(a)))
    if function(b) == 0:
      return (initial_midpoint, step_count, b, b - a, abs(function(b)))

    if (function(a) * function(midpoint) <= 0):
      b = midpoint
    else:
      a = midpoint

    step_count += 1
    current_precision /= 2
    midpoint = (a + b) / 2

  return (initial_midpoint, step_count, midpoint, b - a, abs(function(midpoint)))

def newtons_method(function, first_derivative, a, b, precision):
  x_0 = a
  x_1 = b
  step_count = 0

  while abs(x_1 - x_0) > precision:
    x_0 = x_1
    x_1 = x_0 - function(x_0) / first_derivative(x_0)
    step_count += 1

  return (b - 1, step_count, x_1, abs(x_1 - x_0), abs(function(x_1)))

def modified_newtons_method(function, first_derivative, a, b, precision):
  x_0 = a
  x_1 = b
  denominator = first_derivative(b)
  step_count = 0

  while abs(x_1 - x_0) > precision:
    x_0 = x_1
    x_1 = x_0 - function(x_0) / denominator
    step_count += 1

  return (b - 1, step_count, x_1, abs(x_1 - x_0), abs(function(x_1)))

def secant_method(function, a, b, precision):
  x_0 = a
  x_1 = b
  step_count = 0

  while abs(x_1 - x_0) > precision:
    next = x_1 - function(x_1) / (function(x_1) - function(x_0)) * (x_1 - x_0)
    x_0 = x_1
    x_1 = next
    step_count += 1

  return (a, step_count, x_1, abs(x_1 - x_0), abs(function(x_1)))


def main():
  print("Numerical methods for solving nonlinear equations")
  should_repeat = True

  while should_repeat:
    left_end = float(input("Enter left end of interval: "))
    right_end = float(input("Enter right end of interval: "))
    precision = float(input("Enter target precision: "))
    print(f'Find all odd multiplicity roots of f(x) = 1.2x^4 + 2x^3 - 13x^2 - 14.2x - 24.1 on [{left_end}, {right_end}] with precision = {precision}')
    division_count = int(input("Enter number of division intervals: "))
    print()

    intervals_of_sign_change = get_sign_change_intervals(function, left_end, right_end, division_count)
    print(f'Found {len(intervals_of_sign_change)} intervals of sign change:')
    for interval in intervals_of_sign_change:
      print(interval)
    print()

    for [a, b] in intervals_of_sign_change:
      print(f'Interval {[a, b]}:')
      print_method_result("Bisection Method        ", bisection_method(function, a, b, precision))
      print_method_result("Newton's Method         ", newtons_method(function, first_derivative, a, b, precision))
      print_method_result("Modified Newton's Method", modified_newtons_method(function, first_derivative, a, b, precision))
      print_method_result("Secant Method           ", secant_method(function, a, b, precision))
      print()

    if input("Type 'y' run program one more time: ") != "y":
      should_repeat = False

    print("=============================================================================================")

if __name__ == '__main__':
  main()