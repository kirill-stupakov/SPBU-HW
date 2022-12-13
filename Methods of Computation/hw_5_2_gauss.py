import math
import scipy.integrate as integ
from hw_1 import get_sign_change_intervals, bisection_method

def f(x):
  return math.sqrt(x) * math.sin(x ** 2)

legandre_polynomial_cache = {}
legandre_polynomial_cache[0] = lambda x: 1
legandre_polynomial_cache[1] = lambda x: x

def get_legandre_polynomial(index):
  if index not in legandre_polynomial_cache:
    legandre_polynomial_cache[index] = lambda x: (2 * index - 1) / index * get_legandre_polynomial(index - 1)(x) * x \
    - (index - 1) / index * get_legandre_polynomial(index - 2)(x)

  return legandre_polynomial_cache[index]

def get_anchor_points(count):
  polynomial = get_legandre_polynomial(count)
  signs = get_sign_change_intervals(polynomial, -1, 1, count * 2)
  roots = map(lambda interval: bisection_method(polynomial, interval[0], interval[1], 1e-16)[2], signs)
  return list(roots)

def get_coefficients(anchor_points):
  count = len(anchor_points)
  coefficients = map(lambda x: 2 * (1 - x ** 2) / (count ** 2 * get_legandre_polynomial(count - 1)(x) ** 2), anchor_points)
  return list(coefficients)

def integrate_gauss(a, b, anchor_points):
  print(f'Узлы: {anchor_points}')
  coefficients = get_coefficients(anchor_points)
  print(f'Коэффициенты: {coefficients}')

  sum = 0
  for i in range(len(coefficients)):
    sum += (b - a) / 2 * coefficients[i] * f((b - a) / 2 * anchor_points[i] + (b + a) / 2)
  return sum

def main():
  print('Приближённое вычисление интегралов при помощи КФ Гаусса и Мелера')

  should_repeat = True
  while should_repeat:
    a = float(input('Левый предел интегрирования: '))
    b = float(input('Правый предел интегрирования: '))
    powers = list(map(int, input('Количества узлов: ').split()))

    actual = integ.quad(f, a, b)[0]

    for i in range(len(powers)):
      print(f'========================| N = {powers[i]} |===========================')
      anchor_points = get_anchor_points(powers[i])
      computed_gauss = integrate_gauss(a, b, anchor_points)
      print(f'"Точное" значение: {actual}')
      print(f'КФ Гаусса: {computed_gauss}')
      print(f'Погрешность: {abs(computed_gauss - actual)}')
      print('\n')

    should_repeat = input('Хотите ли ввести значения заново (y / n): ') == 'y'

if __name__ == "__main__":
  main()
