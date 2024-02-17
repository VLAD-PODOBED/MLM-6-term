import pandas as pd

data = pd.read_csv('train.csv')
print(data)

import matplotlib.pyplot as plt

data['Age'].hist()
plt.show()

median = data['Fare'].median()
mean = data['Fare'].mean()
print('Медиана:', median)
print('Среднее значение:', mean)

data.boxplot(column='Age')
plt.show()

description = data['Age'].describe()
print(description)

grouped_data = data.groupby('Sex')
grouped_data['Age'].mean() # среднее значение параметра в каждой группе
grouped_data['Age'].max() # максимальное значение параметра в каждой группе
print(grouped_data['Age'].mean())
print(grouped_data['Age'].max())



