import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Шаг 2: Загрузка данных и обработка пропусков
data = pd.read_csv("train.csv")

# Заполнение пропущенных значений
data["Age"].fillna(data["Age"].mean(), inplace=True)
data.dropna(subset=["Embarked"], inplace=True)

# Замена категориальных признаков на числовые
data_encoded = pd.get_dummies(data)

# Шаг 3: Выделение меток и признаков
Y = data_encoded["Survived"]
X = data_encoded.drop("Survived", axis=1)

# Шаг 4: Разделение набора данных на обучающую и тестовую выборки
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

print("Размер обучающей выборки (X_train):", X_train.shape)
print("Размер тестовой выборки (X_test):", X_test.shape)
print("Размер обучающей выборки (Y_train):", Y_train.shape)
print("Размер тестовой выборки (Y_test):", Y_test.shape)

# Шаг 5: Обучение моделей и расчет точности
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, Y_train)
dt_predictions = dt_model.predict(X_test)
dt_accuracy = accuracy_score(Y_test, dt_predictions)

knn_model = KNeighborsClassifier()
knn_model.fit(X_train, Y_train)
knn_predictions = knn_model.predict(X_test)
knn_accuracy = accuracy_score(Y_test, knn_predictions)

print("Точность модели дерева решений:", dt_accuracy)
print("Точность модели k-ближайших соседей:", knn_accuracy)

# Шаг 6: Подбор наилучших параметров моделей
dt_params = {
    'max_depth': range(1, 10)
}
dt_grid = GridSearchCV(dt_model, dt_params)
dt_grid.fit(X_train, Y_train)
best_dt_model = dt_grid.best_estimator_

knn_params = {
    'n_neighbors': range(1, 10)
}
knn_grid = GridSearchCV(knn_model, knn_params)
knn_grid.fit(X_train, Y_train)
best_knn_model = knn_grid.best_estimator_

print("Лучшие параметры для модели дерева решений:", dt_grid.best_params_)
print("Лучшая модель дерева решений:", best_dt_model)
print("")

print("Лучшие параметры для модели k-ближайших соседей:", knn_grid.best_params_)
print("Лучшая модель k-ближайших соседей:", best_knn_model)


# Шаг 7: Расчет матрицы ошибок
dt_confusion_matrix = confusion_matrix(Y_test, dt_predictions)
knn_confusion_matrix = confusion_matrix(Y_test, knn_predictions)

print("Матрица ошибок для модели дерева решений:")
print(dt_confusion_matrix)
print("")

print("Матрица ошибок для модели k-ближайших соседей:")
print(knn_confusion_matrix)

# Шаг 8: Выбор лучшей модели
if dt_accuracy > knn_accuracy:
    best_model = best_dt_model
    print("Лучшая модель: модель дерева решений")
else:
    best_model = best_knn_model
    print("Лучшая модель: модель k-ближайших соседей")


# Шаг 9: Визуализация модели дерева решений
plt.figure(figsize=(10, 6))
plot_tree(best_dt_model, max_depth=3, feature_names=X.columns, class_names=["Not Survived", "Survived"], filled=True)
plt.show()
