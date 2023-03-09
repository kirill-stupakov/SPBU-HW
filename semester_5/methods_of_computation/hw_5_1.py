import math
import scipy.integrate as integ
import numpy as np
from numpy.polynomial import Polynomial
from hw_1 import get_sign_change_intervals, bisection_method

def f(x):
  return math.sin(x)

def p(x):
  return math.sqrt(1 - x)

mu_cache = {}

def get_mu(index, a, b):
  if index not in mu_cache:
    mu_cache[index] = integ.quad(lambda x: x ** index * p(x), a, b)[0]
  return mu_cache[index]

def get_a(a, b, anchor_points):
  print(f'Узлы: {anchor_points}')
  a_arr = []
  b_arr = []
  for i in range(len(anchor_points)):
    row = list(map(lambda x: x ** i, anchor_points))
    a_arr.append(row)
    b_arr.append(get_mu(i, a, b))

  a = np.array(a_arr)
  b = np.array(b_arr)
  coefficients = np.linalg.solve(a, b).tolist()
  print(f'Коэфициенты: {coefficients}')
  return coefficients

def integrate(a, b, anchor_points):
  q = get_a(a, b, anchor_points)
  ans = 0
  for i in range(len(anchor_points)):
    ans += q[i] * f(anchor_points[i])
  return ans

def get_orthagonal_polynomial_coefficients(degree, a, b):
  a_arr = []
  b_arr = []
  for i in range(degree):
    row = []
    for j in range(degree):
      row.append(get_mu(degree - 1 + i - j, a, b))
    a_arr.append(row)
    b_arr.append(-get_mu(degree + i, a, b))

  a = np.array(a_arr)
  b = np.array(b_arr)
  coefficients = np.linalg.solve(a, b).tolist()
  coefficients.insert(0, 1)
  return coefficients

def get_anchor_points(count, a, b):
  coefficients = get_orthagonal_polynomial_coefficients(count, a, b)
  print('Ортогональный многочлен: ', end='')
  polynomial = Polynomial(coefficients[::-1])
  print(polynomial)
  signs = get_sign_change_intervals(polynomial, a, b, count * 2)
  roots = map(lambda interval: bisection_method(polynomial, interval[0], interval[1], 1e-16)[2], signs)
  return list(roots)

def main():
  print('Приближённое вычисление интегралов при помощи КФ НАСТ')

  should_repeat = True
  while should_repeat:
    a = float(input('Левый предел интегрирования: '))
    b = float(input('Правый предел интегрирования: '))
    n = int(input('Количество узлов: '))

    actual = integ.quad(lambda x: f(x) * p(x), a, b)[0]
    print('===============================================================')
    anchor_points = get_anchor_points(n, a, b)
    computed = integrate(a, b, anchor_points)
    print('===============================================================')

    print(f'"Точное" значение: {actual}')
    print(f'Посчитанное значение: {computed}')
    print(f'Погрешность: {abs(computed - actual)}')

    should_repeat = input('Хотите ли ввести значения заново (y / n): ') == 'y'

if __name__ == "__main__":
  main()
