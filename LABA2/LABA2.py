import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('smartwatches.csv')
print("Размер исходных данных: ", data.shape)

# 1. Выявление пропусков данных
# Визуальный способ
plt.figure(figsize=(10, 6))
plt.title('Супер-пупер данные')
sns.heatmap(data.isnull(), cbar=False, cmap='viridis')
plt.show()

# Расчетный способ
missing_data = data.isnull().sum()
missing_data_percent = (missing_data / len(data)) * 100
missing_data_table = pd.concat([missing_data, missing_data_percent], axis=1)
missing_data_table = missing_data_table.rename(columns={0: 'Missing Data Count', 1: 'Missing Data Percentage'})
print(missing_data_table)

# 2. Исключение строк и столбцов с наибольшим количеством пропусков
# Удаление строк с наибольшим количеством пропусков
num_missing = data.isnull().sum(axis=1)  
threshold_rows = int(len(data.columns) * 0.5)  
data_cleaned_rows = data[num_missing <= threshold_rows]

# Удаление столбцов с наибольшим количеством пропусков
threshold_columns = int(len(data) * 0.5)  
data_cleaned_columns = data.dropna(thresh=threshold_columns, axis=1)

# Отображение информации о новых размерах датасетов
print("Размер данных после удаления строк с пропусками: ", data_cleaned_rows.shape) 
print("Размер данных после удаления столбцов с пропусками: ", data_cleaned_columns.shape)  

# 3. Замена оставшихся пропусков на логически обоснованные значения

numeric_cols = data.select_dtypes(include='number').columns
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())

categorical_cols = data.select_dtypes(include='object').columns
data[categorical_cols] = data[categorical_cols].fillna(data[categorical_cols].mode().iloc[0])

print("Количество оставшихся пропусков: ", data.isnull().sum().sum())

# 4.Постройте гистограмму распределения исходного датасета до и после обработки пропусков.

plt.figure(figsize=(10, 6))
plt.hist(data['Weight'].dropna(), bins=20, color='blue', alpha=0.5, label='Before')
plt.title('Distribution before NaN Handling')
plt.xlabel('Weight')
plt.ylabel('Frequency')
plt.legend()
plt.show()

data_processed = data.dropna(subset=['Weight'])
plt.figure(figsize=(10, 6))
plt.hist(data_processed['Weight'], bins=20, color='green', alpha=0.5, label='After')
plt.title('Distribution after NaN Handling')
plt.xlabel('Weight')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# 5. Проверка наличия выбросов и удаление аномальных записей

Q1 = data['Rating'].quantile(0.25)
Q3 = data['Rating'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
data_filtered = data[(data['Rating'] >= lower_bound) & (data['Rating'] <= upper_bound)]
num_outliers = len(data) - len(data_filtered)
print("Количество удаленных аномальных записей: ", num_outliers)

# 6. Приведение параметров к числовому виду (кодирование текстовых данных)
categorical_cols = data.select_dtypes(include='object').columns
data[categorical_cols] = data[categorical_cols].apply(lambda x: pd.factorize(x)[0])

# 7. Сохранение обработанных данных в новый файл
data.to_csv('dataset_encoded.csv', index=False)


