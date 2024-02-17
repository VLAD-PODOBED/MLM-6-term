import numpy as np

#NumPy-------------------------
# Создание двумерного массива из 20 случайных целых чисел
array = np.random.randint(0, 10, (4, 5))  # Форма (4, 5) для двумерного массива 4x5
print("Исходный массив:")
print(array)
# Разделение двумерного массива на 2 равных части
split_arrays = np.split(array, 2)
array1, array2 = split_arrays[0], split_arrays[1]

print("\nПервый массив:")
print(array1)
print("\nВторой массив:")
print(array2)
# Найти все значения, например равные 6, в первом массиве
desired_value = 6
matching_elements = np.where(array1 == desired_value)

print(f"\nНайденные значения равные {desired_value} в первом массиве:")
print(array1[matching_elements])

# Подсчет количества найденных элементов
count_matches = len(matching_elements[0])
print(f"\nКоличество найденных элементов равных {desired_value}: {count_matches}")

#Pandas---------------------------
import numpy as np
import pandas as pd

# Создание объекта Series из массива NumPy
np_array = np.array([10, 20, 30, 40, 50])
series = pd.Series(np_array)

# Выполнение различных математических операций с объектом Series
print("Объект Series:")
print(series)
print("\nМатематические операции:")
print("Сумма всех элементов:", series.sum())
print("Среднее значение элементов:", series.mean())
print("Максимальное значение:", series.max())
print("Минимальное значение:", series.min())

# Создание объекта DataFrame из массива NumPy
np_array_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
df = pd.DataFrame(np_array_2d)

# Написание строки заголовков для DataFrame
df.columns = ['A', 'B', 'C']
print("\nОбъект DataFrame с заголовками:")
print(df)

# Удаление строки с индексом 1
df = df.drop(1)
print("\nDataFrame после удаления строки с индексом 1:")
print(df)

# Удаление столбца 'B'
df = df.drop('B', axis=1)
print("\nDataFrame после удаления столбца 'B':")
print(df)

# Вывод размера DataFrame
print("\nРазмер DataFrame (строки, столбцы):", df.shape)

# Найти все элементы, равные числу 3 в DataFrame
desired_value = 3
matching_elements = df[df == desired_value]
print(f"\nНайденные значения равные {desired_value}:")
print(matching_elements)
