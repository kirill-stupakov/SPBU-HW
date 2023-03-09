import math
from hw_1 import *
from hw_2 import *

def function(x):
  return math.cos(x) + 2 * x
  # return math.exp(-x) - x**2 / 2


def main():
  print('Задача обратного интерполирования f(x) = cos(x) + 2x')

  left_end = float(input('Введите левый конец отрезка: '))
  right_end = float(input('Введите правый конец отрезка: '))
  lookup_entry_count = int(input('Введите число значений в таблице: '))
  initial_lookup_table = create_lookup_table(function, lookup_entry_count, left_end, right_end)
  print_lookup_table(initial_lookup_table, '===| Исходная таблица значений |===')

  should_repeat = True
  while should_repeat:
    point = float(input('Введите искомое значение: '))
    power = input_until_valid(f'Введите степень интерполяционного многочлена (<= {lookup_entry_count - 1}): ', 'Введено недопустимое значение. Повторите: ', lambda x: x <= lookup_entry_count - 1)
    inverted_lookup_table = initial_lookup_table[::-1]
    sorted_lookup_table = sort_lookup_table_by_distance(inverted_lookup_table, point, power + 1)
    print_lookup_table(sorted_lookup_table, '===| Отсортированная таблица значений |===')

    print('===| Первый способ |===')
    polynomial = newton_polynomial(sorted_lookup_table)
    print(f'Значение аргумента: {polynomial(point)}')
    print(f'Модуль невязки: {abs(function(polynomial(point)) - point)}')

    print('===| Второй способ |===')
    polynomial_2 = newton_polynomial(initial_lookup_table) - Polynomial([point])
    precision = float(input("Введите требуемую точность: "))
    division_count = int(input("Введите количество интервалов разбиения: "))
    intervals_of_sign_change = get_sign_change_intervals(polynomial_2, left_end, right_end, division_count)

    for [a, b] in intervals_of_sign_change:
      print(f'Интервал {[a, b]}:')
      print_method_result("Метод секущих", secant_method(polynomial_2, a, b, precision))

    should_repeat = input('Хотите ли ввести значения заново (y / n): ') == 'y'

if __name__ == "__main__":
    main()
