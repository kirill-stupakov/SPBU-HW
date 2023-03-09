import math
import numpy as np
import scipy.integrate as integ

def f(x):
  return math.sin(x)

def p(x):
  return math.pow(1 - x, 3 / 4)

def get_mu(k, a, b):
  return integ.quad(lambda x: (x ** k) * p(x), a, b)[0]

def get_a(a, b, anchor_points):
  a_arr = []
  b_arr = []
  for i in range(len(anchor_points)):
    row = list(map(lambda x: x ** i, anchor_points))
    a_arr.append(row)
    b_arr.append(get_mu(i, a, b))

  a = np.array(a_arr)
  b = np.array(b_arr)
  x = np.linalg.solve(a, b)
  return x

def integrate(a, b, anchor_points):
  q = get_a(a, b, anchor_points)
  ans = 0
  for i in range(len(anchor_points)):
    ans += q[i] * f(anchor_points[i])
  return ans

def main():
  print('Нахождение производных таблично-заданной функции exp(1.5x) по формулам численного дифференцирования')

  should_repeat = True
  while should_repeat:
    a = float(input('Левый предел интегрирования: '))
    b = float(input('Правый предел интегрирования: '))
    anchor_points = [1/4, 3/4]

    actual = integ.quad(lambda x: f(x) * p(x), a, b)[0]
    computed = integrate(a, b, anchor_points)

    print(f'"Точное" значение: {actual}')
    print(f'Посчитанное значение: {computed}')
    print(f'Погрешность: {abs(computed - actual)}')

    should_repeat = input('Хотите ли ввести значения заново (y / n): ') == 'y'

if __name__ == "__main__":
    main()