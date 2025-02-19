import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

#-----------------------------------------------------------------------------------------------------------------------------------

def weight_factor(data, swap):  # нахождение весовых коэффициентов или же параметров для нашей регрессионной прямой
    sum_of_x = 0 # значение суммы от 0 до n по Xi
    sum_of_y = 0 # значение суммы от 0 до n по Yi
    sum_of_xy = 0 # значение суммы от 0 до n по XiYi
    sum_of_x2 = 0 # значение суммы от 0 до n по Xi ^ 2
    n = len(data) # количество точек
    for i in range(n):
        x = data[i][0] # координата X i-ой точки
        y = data[i][1] # координата Y i-ой точки
        if swap: # если значения столбцов X и Y поменять местами
            x, y = y, x
        sum_of_x += x
        sum_of_y += y
        sum_of_xy += x * y
        sum_of_x2 += x ** 2
    b = (sum_of_x * sum_of_y / n - sum_of_xy) / (sum_of_x ** 2 / n - sum_of_x2) # формулы для a и b взяты из лекции по линейной регрессии на плоскости
    a = (sum_of_y - b * sum_of_x) / n
    return (a, b)

def regression_line(x, a, b): # функция регрессионной прямой
    return a + b * x

#-----------------------------------------------------------------------------------------------------------------------------------

# Вводимые с клавиатуры данные (первый аргумент - путь к файлу; второй аргумент (при наличии) - показатель того, нужно ли менять X и Y местами (0 - менять не нужно (стандарт); 1- менять нужно))
path_to_file = sys.argv[1] # путь к папке с данными
swap = False # показатель того, нужно ли менять столбцы X и Y местами
if (len(sys.argv) > 2): 
    if (sys.argv[2] == 1): # если столбцы X и Y нужно поменять местами
        swap = True
data = pd.read_csv(path_to_file) # читаем файл

np_array = data.to_numpy() # конвертируем DataFrame в NumPy array
a, b = weight_factor(np_array, swap) # высчитываем коэффициенты a и b для регресионной прямой
X, Y = data.columns[0], data.columns[1] # обозначаем какая из колонок в файле является столбцом X, а какая столбцом Y (по станадрту 1-ая колонка - X, менять местами будем позже)
transposed = np.transpose(np_array)  # траснпонируем, т.к. на данный момент каждый массив в np_array это строка в исходном файле, а нужно чтобы массивами были столбцы
values = transposed[0] # координаты X (первый столбец)
if swap:  # счиытваем была ли замена параметров местами
    X, Y = Y, X # меняем X и Y местами
    values = transposed[1] # координаты X (теперь это второй столбец)

data.plot(x=X, y=Y, kind="scatter", color="#32CD32")  # ставим на системе координат точки из файла
ax = plt.subplot() # создаем пустой подграфик, который будет отображать регрессионную прямую и метод наименьших квадратов
ax.plot(values, regression_line(values, a, b), color="#000000")  # строим регрессионную прямую на нашем подграфике
for i in range(len(values)):  # метод наименьших квадратов
    x = values[i] # координата X
    if swap:
        y = transposed[0][i] # координата Y при замене местами столбцов
    else:
        y = transposed[1][i] # координат Y при отсутствии замены местами столбцов
    fx = regression_line(x, a, b) # значение приблизительной функции, по которой строилась регрессионная прямая
    loss = abs(y - fx) # потеря/отклонение прбилзительного значения от настоящего
    if min(y, fx) == y:
        ax.add_patch(Rectangle((x, min(y, fx)), loss, loss, edgecolor="#FF0000", facecolor="#F08080")) # рисуем наименьшие квадраты, если настоящее значение меньше приблизительного
    else:
        ax.add_patch(Rectangle((x - loss, min(y, fx)), loss, loss, edgecolor="#FF0000", facecolor="#F08080")) # рисуем наименьшие квадраты, если настоящее значение больше приблизительного
plt.xlabel(f"Hours (count: {data.count(axis=0).iloc[0]}; min: {data.min(axis=0).iloc[0]}; max: {data.max(axis=0).iloc[0]}; mean: {data.mean(axis=0).iloc[0]})") # выводим статистику по столбцу Hours
plt.ylabel(f"Scores (count: {data.count(axis=0).iloc[1]}; min: {data.min(axis=0).iloc[1]}; max: {data.max(axis=0).iloc[1]}; mean: {data.mean(axis=0).iloc[1]})") # выводим статистику по столбцу Scores

plt.show() # выводим итоговый график 