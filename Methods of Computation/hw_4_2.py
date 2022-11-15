import math
import scipy.integrate as integrate
from prettytable import PrettyTable

def func(x):
  any = math.sin(x) * math.pow(1 - x, 3 / 4)
  zero_deg = 4
  first_deg = 2 * x + 1
  second_deg = 3 * x ** 2 + 1.5 * x - 2
  third_deg = -1 * x ** 3 + 4 * x ** 2 - 1.5 * x + 3
  return any

def left_rect(a, b, f):
  return f(a) * (b - a)

def right_rect(a, b, f):
  return f(b) * (b - a)

def middle_rect(a, b, f):
  return f((a + b) / 2) * (b - a)

def trapezoid(a, b, f):
  return (f(a) + f(b)) * (b - a) / 2

def simpson(a, b, f):
  return (f(a) + 4 * f((a + b) / 2) + f(b)) * (b - a) / 6

def three_eights(a, b, f):
  h = (b - a) / 3
  return (1/8 * f(a) + 3/8 * f(a + h) + 3/8 * f(a + 2 * h) + 1/8 * f(b)) * (b - a)

def parse_result(name, computed, actual):
  return [name, computed, abs(computed - actual), f'{abs((computed - actual) / actual) * 100} %']

def get_rows(names, formulas, a, b, actual):
  res = []
  for i in range(len(formulas)):
    res.append(parse_result(names[i], formulas[i](a, b, func), actual))
  return res

def main():
  print('Нахождение производных таблично-заданной функции exp(1.5x) по формулам численного дифференцирования')

  should_repeat = True
  while should_repeat:
    a = float(input('Левый предел интегрирования: '))
    b = float(input('Правый предел интегрирования: '))
    
    actual = integrate.quad(func, a, b)[0]
    header = ['Метод', 'Значение', 'Абс. погрешность', 'Отн. погрешность']
    table = PrettyTable(header)
    names = ['Л.П.', 'П.П.', 'С.П.', 'Трапеции', 'Симпсона', '3/8']
    formulas = [left_rect, right_rect, middle_rect, trapezoid, simpson, three_eights]
    table.add_rows(get_rows(names, formulas, a, b, actual))

    print()
    print(f'Точное значение: {actual}')
    print(table)

    should_repeat = input('Хотите ли ввести значения заново (y / n): ') == 'y'

if __name__ == "__main__":
    main()