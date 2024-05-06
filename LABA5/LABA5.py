import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
from sklearn.metrics import r2_score

# Прочитаем данные из файла CSV
data = pd.read_csv('train.csv')
data = data.drop(columns=['Id'])
data = pd.get_dummies(data)
data = data.fillna(data.mean())

# Рассчитаем матрицу корреляций
correlation_matrix = data.corr()

correlation_coefficient = data["OverallQual"].corr(data["SalePrice"])
print(correlation_coefficient)

# 2 Создадим тепловую карту
#plt.figure(figsize=(12, 10))
#sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
#plt.title('Матрица корреляций')
#plt.show()

# 3: Построение матрицы диаграмм рассеяния
selected_vars = ["OverallQual", "GrLivArea", "GarageArea", "YearBuilt", "SalePrice"]
subset_data = data[selected_vars]
sns.pairplot(subset_data)
plt.show()

# 4: Выбор переменных для простой линейной регрессии
# выбираем "OverallQual", "GrLivArea" и "GarageArea" как параметры для модели

# 5: Расчет и визуализация модели на диаграмме рассеяния
X = data[["OverallQual", "GrLivArea", "GarageArea"]]
y = data["SalePrice"]
model = LinearRegression()
model.fit(X, y)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X["OverallQual"], X["GrLivArea"], y)
ax.plot_trisurf(X["OverallQual"], X["GrLivArea"], model.predict(X), color='red', alpha=0.5)
ax.set_xlabel("OverallQual")
ax.set_ylabel("GrLivArea")
ax.set_zlabel("SalePrice")
plt.show()

# 6: Оценка качества модели
y_pred = model.predict(X)
mse = mean_squared_error(y, y_pred)
print("Mean Squared Error:", mse)

y_pred = model.predict(X)
r2 = r2_score(y, y_pred)

print("R-Squared Score:", r2)

# 7: Добавление дополнительных параметров в модель
X = data[["OverallQual", "GrLivArea", "GarageArea", "YearBuilt", "LotArea"]]
model_multiple = LinearRegression()
model_multiple.fit(X, y)

# 8: Расчет и оценка модели с несколькими параметрами
y_pred_multiple = model_multiple.predict(X)
mse_multiple = mean_squared_error(y, y_pred_multiple)
print("Mean Squared Error (Multiple Variables):", mse_multiple)

# 9: Вывод о наилучшей модели
if mse < mse_multiple:
    print("Простая линейная регрессия лучше описывает зависимость целевой переменной от параметров.")
else:
    print("Модель с несколькими параметрами лучше описывает зависимость целевой переменной от параметров.")