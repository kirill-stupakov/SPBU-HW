import math
from numpy.polynomial import Polynomial

def function(x):
    return math.cos(x) + 2 * x
    # return 0.5 * x**3 - 0.26 * x + 2.3

def create_lookup_table(func, rows, start, end):
    result = [[], []]
    multiplier = (end - start) / (rows - 1)
    for i in range(rows):
        result[0].append(start + i * multiplier)
        result[1].append(func(result[0][i]))

    return result

def print_lookup_table(table, message):
    print()
    print(message)
    for i in range(len(table[0])):
        print(f'f({table[0][i]}) = {table[1][i]}')
    print()

def input_until_valid(message, invalid_message, validator):
    result = int(input(message))
    while (not validator(result)):
        result = int(input(invalid_message))

    return result

def sort_lookup_table_by_distance(table, point, limit):
    nodes = []
    for i in range(len(table[0])):
        nodes.append([table[0][i], table[1][i]])

    nodes.sort(key=lambda x: abs(x[0] - point))

    result = [[], []]
    for node in nodes[:limit]:
        result[0].append(node[0])
        result[1].append(node[1])

    return result

def lp_denominator(k, lookup_table):
    result = 1
    for i in range(len(lookup_table[0])):
        if (i != k):
            result *= lookup_table[0][k] - lookup_table[0][i]

    return result

def lagrange_polynomial(lookup_table, t):
    power = len(lookup_table[0])
    result = Polynomial([0])

    for i in range(power):
        numerator = Polynomial([1])
        for j in range(power):
            if (i != j):
                numerator *= Polynomial([-lookup_table[0][j], 1])
        
        result += numerator * lookup_table[1][i] / lp_denominator(i, lookup_table)

    return result

def _lagrange_polynomial(lookup_table, t):
    power = len(lookup_table[0])
    # result = Polynomial([0])
    result = 0

    for i in range(power):
        numerator = 1
        for j in range(power):
            if (i != j):
                # numerator *= Polynomial([-lookup_table[0][j], 1])
                numerator *= t - lookup_table[0][j]
        
        result += numerator * lookup_table[1][i] / lp_denominator(i, lookup_table)

    return result

def newton_coefficients(lookup_table):
    power = len(lookup_table[0])
    divided_sum = lookup_table.copy()

    for current_column in range(2, power + 1):
        new_column = []
        
        for current_row in range(power - current_column + 1):
            numerator = divided_sum[current_column - 1][current_row + 1] - divided_sum[current_column - 1][current_row]
            denominator = divided_sum[0][current_row + current_column - 1] - divided_sum[0][current_row]
            new_column.append(numerator / denominator)
        
        divided_sum.append(new_column)

    return list(map(lambda x: x[0], divided_sum[1:]))
        

def newton_polynomial(lookup_table):
    coefficients = newton_coefficients(lookup_table)
    result = Polynomial([0])
    
    for current_power in range(len(coefficients)):
        add_polynomial = Polynomial([coefficients[current_power]])
        for i in range(current_power):
            add_polynomial *= Polynomial([-lookup_table[0][i], 1])

        result += add_polynomial

    return result

def newton_polynomial(lookup_table):
    coefficients = newton_coefficients(lookup_table)
    result = Polynomial([0])

    for current_power in range(len(coefficients)):
        add_polynomial = Polynomial([coefficients[current_power]])
        for i in range(current_power):
            add_polynomial *= Polynomial([-lookup_table[0][i], 1])

        result += add_polynomial

    return result


def main():
    print('Задача алгебраического интерполирования f(x) = cos(x) + 2x')

    left_end = float(input('Введите левый конец отрезка: '))
    right_end = float(input('Введите правый конец отрезка: '))
    lookup_entry_count = int(input('Введите число значений в таблице: '))
    initial_lookup_table = create_lookup_table(function, lookup_entry_count, left_end, right_end)
    print_lookup_table(initial_lookup_table, '===| Исходная таблица значений |===')

    should_repeat = True
    while should_repeat:
        point = float(input('Введите точку интерполирования: '))
        power = input_until_valid(f'Введите степень интерполяционного многочлена (<= {lookup_entry_count - 1}): ', 'Введено недопустимое значение. Повторите: ', lambda x: x <= lookup_entry_count - 1)
        sorted_lookup_table = sort_lookup_table_by_distance(initial_lookup_table, point, power + 1)
        print_lookup_table(sorted_lookup_table, '===| Отсортированная таблица значений |===')

        # lagrange = lagrange_polynomial(sorted_lookup_table, point)
        # print(f'Многочлен Лагранжа: {lagrange}')
        # print(f'Значение: {lagrange(point)}')
        # print(f'Погрешность: {abs(lagrange(point) - function(point))}')

        lagrange = _lagrange_polynomial(sorted_lookup_table, point)
        print(f'Многочлен Лагранжа')
        print(f'Значение: {lagrange}')
        print(f'Погрешность: {abs(lagrange - function(point))}')

        print('\n===============================\n')

        newton = newton_polynomial(sorted_lookup_table)
        print(f'Многочлен Ньютона: {newton}')
        print(f'Значение: {newton(point)}')
        print(f'Погрешность: {abs(newton(point) - function(point))}')

        should_repeat = input('Хотите ли ввести значения заново (y / n): ') == 'y'

if __name__ == "__main__":
    main()
