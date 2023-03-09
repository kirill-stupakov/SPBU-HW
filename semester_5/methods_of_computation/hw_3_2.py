import math
from prettytable import PrettyTable
from hw_2 import *

def function(x):
  return math.exp(1.5 * x)

def first_derivative(x):
  return 1.5 * math.exp(1.5 * x)

def second_derivative(x):
  return 1.5 * 1.5 * math.exp(1.5 * x)

def f_d(lookup_table, lookup_step, index):
  if (index == 0):
    return (-3 * lookup_table[1][0] + 4 * lookup_table[1][1] - lookup_table[1][2]) / (2 * lookup_step)
  
  if (index == len(lookup_table[0]) - 1):
    return (3 * lookup_table[1][index] - 4 * lookup_table[1][index - 1] + lookup_table[1][index - 2]) / (2 * lookup_step)

  return (lookup_table[1][index + 1] - lookup_table[1][index - 1]) / (2 * lookup_step)

def s_d(lookup_table, lookup_step, index):
  if index == 0 or index == len(lookup_table[0]) - 1:
    return math.nan

  return (lookup_table[1][index + 1] - 2 * lookup_table[1][index] + lookup_table[1][index - 1]) / (lookup_step ** 2)

def main():
  print('Нахождение производных таблично-заданной функции exp(1.5x) по формулам численного дифференцирования')

  should_repeat = True
  while should_repeat:
    left_end = float(input('Введите левый конец отрезка: '))
    lookup_entry_count = int(input('Введите число значений в таблице: '))
    lookup_step = float(input('Введите шаг таблицы: '))
    lookup_table = create_lookup_table(function, lookup_entry_count, left_end, left_end + (lookup_entry_count - 1) * lookup_step)
    # print_lookup_table(lookup_table, '===| Исходная таблица значений |===')

    table = PrettyTable(['x_i', 'f(x_i)', 'f\'(x_i)', 'абс. погрешность f\'', 'отн. погрешность f\'', 'f\'\'(x_i)', 'абс. погрешность f\'\'', 'отн. погрешность f\'\''])

    for i in range(len(lookup_table[0])):
      x_i = lookup_table[0][i]
      value = lookup_table[1][i]

      actual_first = first_derivative(x_i)
      first = f_d(lookup_table, lookup_step, i)
      first_abs = abs(first - actual_first)
      first_rel = f'{first_abs / actual_first * 100} %'

      actual_second = second_derivative(x_i)
      second = s_d(lookup_table, lookup_step, i)
      second_abs = abs(second - actual_second)
      second_rel = f'{second_abs / actual_second * 100} %'

      table.add_row([x_i, value, first, first_abs, first_rel, second, second_abs, second_rel])

    print(table)

    should_repeat = input('Хотите ли ввести значения заново (y / n): ') == 'y'

if __name__ == "__main__":
    main()
