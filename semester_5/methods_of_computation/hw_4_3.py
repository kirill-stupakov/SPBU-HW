import scipy.integrate as integrate
from prettytable import PrettyTable

from  hw_4_2 import *

def func(x):
  return math.exp(x)

def make_compound(quad_formula):
  return lambda a, b, m, f: sum([quad_formula(a + i * (b - a)/m, a + (i + 1) * (b - a)/m, f) for i in range(m)])

def parse_result(name, computed, actual, expected_diff):
  return [name, computed, abs(computed - actual), f'{abs((computed - actual) / actual) * 100} %', expected_diff]

def get_rows(names, formulas, expected_diffs, a, b, m, actual):
  res = []
  for i in range(len(formulas)):
    res.append(parse_result(names[i], formulas[i](a, b, m, func), actual, expected_diffs[i](a, b, m)))
  return res

def left_rect_diff(a, b, m):
  return (b - a) ** 2 / (2 * m) * func(b)

def right_rect_diff(a, b, m):
  return left_rect_diff(a, b, m)

def middle_rect_diff(a, b, m):
  return (b - a) ** 3 / (24 * m ** 2) * func(b)

def trapezoid_diff(a, b, m):
  return (b - a) ** 3 / (12 * m ** 2) * func(b)

def simpson_diff(a, b, m):
  return (b - a) ** 5 / (1880 * m ** 4) * func(b)

def main():
  print('Приближённое вычисление интеграла по составным квадратурным формулам')

  should_repeat = True
  while should_repeat:
    a = float(input('Левый предел интегрирования: '))
    b = float(input('Правый предел интегрирования: '))
    m = int(input('Число промежутков деления: '))
    
    actual = integrate.quad(func, a, b)[0]
    header = ['Метод', 'Значение', 'Абс. погрешность', 'Отн. погрешность', 'Оценка погрешности']
    table = PrettyTable(header)
    names = ['Л.П.', 'П.П.', 'С.П.', 'Трапеции', 'Симпсона']
    formulas = list(map(make_compound, [left_rect, right_rect, middle_rect, trapezoid, simpson]))
    expected_diffs = [left_rect_diff, right_rect_diff, middle_rect_diff, trapezoid_diff, simpson_diff]
    table.add_rows(get_rows(names, formulas, expected_diffs, a, b, m, actual))

    print()
    print(f'Точное значение: {actual}')
    print(table)

    should_repeat = input('Хотите ли ввести значения заново (y / n): ') == 'y'

if __name__ == "__main__":
    main()