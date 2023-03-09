import scipy.integrate as integrate
from prettytable import PrettyTable

from  hw_4_2 import *

def func(x):
  return math.exp(x)

def make_compound(quad_formula):
  return lambda a, b, m, f: sum([quad_formula(a + i * (b - a)/m, a + (i + 1) * (b - a)/m, f) for i in range(m)])

def get_rows(names, formulas, a, b, m, actual):
  res = []
  for i in range(len(formulas)):
    res.append(parse_result(names[i], formulas[i](a, b, m, func), actual))
  return res

def get_runge_rows(names, single_rows, mult_rows, l, actual):
  r = [1, 1, 2, 2, 4]
  res = []
  for i in range(len(single_rows)):
    value = (math.pow(l, r[i]) * mult_rows[i] - single_rows[i]) / (math.pow(l, r[i]) - 1)
    res.append(parse_result(names[i], value, actual))
  return res

def main():
  print('Приближённое вычисление интеграла по составным квадратурным формулам')

  should_repeat = True
  while should_repeat:
    a = float(input('Левый предел интегрирования: '))
    b = float(input('Правый предел интегрирования: '))
    m = int(input('Число промежутков деления: '))
    l = int(input('Множитель: '))
    
    actual = integrate.quad(func, a, b)[0]
    header = ['Метод', 'Значение', 'Абс. погрешность', 'Отн. погрешность']
    table = PrettyTable(header)
    names = ['Л.П.', 'П.П.', 'С.П.', 'Трапеции', 'Симпсона']
    formulas = list(map(make_compound, [left_rect, right_rect, middle_rect, trapezoid, simpson]))
    single_rows = get_rows(names, formulas, a, b, m, actual)
    mult_rows = get_rows(names, formulas, a, b, m * l, actual)
    runge_rows = get_runge_rows(names, list(map(lambda a: a[1], single_rows)), list(map(lambda a: a[1], mult_rows)), l, actual)

    print()
    print(f'Точное значение: {actual}')

    table.add_rows(single_rows)
    print(table)
    table.clear_rows()

    table.add_rows(mult_rows)
    print(table)
    table.clear_rows()

    table.add_rows(runge_rows)
    print(table)
    table.clear_rows()

    should_repeat = input('Хотите ли ввести значения заново (y / n): ') == 'y'

if __name__ == "__main__":
    main()