import math
import scipy.integrate as integ
from hw_5_2_gauss import get_anchor_points, integrate_gauss, get_coefficients

def f(x):
  return math.sin(x) * math.sqrt(1 - x)

def main():
  print('Приближённое вычисление интегралов при помощи составной КФ Гаусса')

  should_repeat = True
  while should_repeat:
    a = float(input('Левый предел интегрирования: '))
    b = float(input('Правый предел интегрирования: '))
    intervals_count = int(input('Количество интервалов разбиения: '))
    anchor_points_count = int(input('Количество узлов: '))

    actual = integ.quad(f, a, b)[0]
    anchor_points = get_anchor_points(anchor_points_count)
    coefficients = get_coefficients(anchor_points)
    print(f'Узлы: {anchor_points}')
    print(f'Коэффициенты: {coefficients}')

    computed = 0
    h = (b - a) / intervals_count

    for i in range(intervals_count):
      for j in range(len(coefficients)):
        computed += h / 2 * coefficients[j] * f(h / 2 * anchor_points[j] + a + (i + 0.5) * h)

    print(f'"Точное" значение: {actual}')
    print(f'СКФ Гаусса: {computed}')
    print(f'Погрешность: {abs(computed - actual)}')
    print('\n')
    
    should_repeat = input('Хотите ли ввести значения заново (y / n): ') == 'y'

if __name__ == "__main__":
  main()
