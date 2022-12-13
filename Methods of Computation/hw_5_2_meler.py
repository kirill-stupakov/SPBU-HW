import math
import scipy.integrate as integ

def f(x):
  return math.cos(x) * (1 + x ** 2)

def p(x):
  return 1 / math.sqrt(1 - x ** 2)

def get_anchor_points(count):
  points = []
  for i in range(count):
    points.append(math.cos(math.pi * (2 * i + 1) / (2 * count)))
  return points

def integrate(anchor_points):
  print(f'Узлы: {anchor_points}')
  print(f'Коэффициент: {math.pi / len(anchor_points)}')

  sum = 0
  for i in range(len(anchor_points)):
    sum += math.pi / len(anchor_points) * f(anchor_points[i])
  return sum

def main():
  print('Приближённое вычисление интегралов при помощи КФ Мелера')

  should_repeat = True
  while should_repeat:
    powers = list(map(int, input('Количества узлов: ').split()))

    # actual = integ.quad(lambda x: f(x) * p(x), -1, 1)[0]
    actual = 3.42541917388465747078036645749

    for i in range(len(powers)):
      print(f'========================| N = {powers[i]} |===========================')
      anchor_points = get_anchor_points(powers[i])
      computed_meler = integrate(anchor_points)
      print(f'"Точное" значение: {actual}')
      print(f'КФ Мелера: {computed_meler}')
      print(f'Погрешность: {abs(computed_meler - actual)}')
      print('\n')

    should_repeat = input('Хотите ли ввести значения заново (y / n): ') == 'y'

if __name__ == "__main__":
  main()
