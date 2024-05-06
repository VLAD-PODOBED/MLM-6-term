import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error,r2_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

data = pd.read_csv("train.csv")

data = pd.get_dummies(data)

imputer = SimpleImputer(strategy='mean')
data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# 2. Обучение модели случайного леса (Random Forest). Оценка точности.
X = data.drop(columns=["SalePrice"])
y = data["SalePrice"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf_model = RandomForestRegressor()
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy = root_mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Среднеквадратичная ошибка модели случайного леса:", accuracy)
print("Точность модели случайного леса (R2):", r2)
#точность

# 3. Уменьшение количества параметров датасета с помощью Feature Selection.
selector = VarianceThreshold(threshold=0.01)
X_train_reduced = selector.fit_transform(X_train)
X_test_reduced = selector.transform(X_test)

# 4. Обучение модели случайного леса на уменьшенном датасете. Оценка точности.
rf_model.fit(X_train_reduced, y_train)
y_pred_reduced = rf_model.predict(X_test_reduced)
accuracy_reduced = root_mean_squared_error(y_test, y_pred_reduced)
r3 = r2_score(y_test, y_pred_reduced)
print("Среднеквадратичная ошибка после сокращения признаков:", accuracy_reduced)
print("Точность модели случайного леса (R2):", r3)

# 5. Применение метода PCA к исходному датасету для нахождения 2 главных компонент.
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)



# 6. Визуализация данных по этим двум компонентам.
plt.scatter(X_pca[:,0], X_pca[:,1])
plt.xlabel('Главная компонента 1')
plt.ylabel('Главная компонента 2')
plt.title('Визуализация PCA')
plt.show()

# 7. Обучение модели случайного леса на полученной модели PCA с двумя компонентами. Оценка точности и времени.
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca, y, test_size=0.2, random_state=42)
rf_model.fit(X_train_pca, y_train_pca)
start_time = time.time()
y_pred_pca = rf_model.predict(X_test_pca)
end_time = time.time()
accuracy_pca = root_mean_squared_error(y_test_pca, y_pred_pca)
print("Среднеквадратичная ошибка после PCA:", accuracy_pca)
print("Время предсказания с использованием PCA:", end_time - start_time)

# 8. Нахождение оптимального количества главных компонент для сохранения 90% дисперсии.
pca = PCA()
pca.fit(X)
cumsum = np.cumsum(pca.explained_variance_ratio_)
d = np.argmax(cumsum >= 0.90) + 1
print("Количество главных компонент для сохранения 90% дисперсии:", d)

# 9. Обучение модели с определенным количеством компонент. Оценка точности и времени.
pca = PCA(n_components=d)
X_pca = pca.fit_transform(X)
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca, y, test_size=0.2, random_state=42)
rf_model.fit(X_train_pca, y_train_pca)
start_time = time.time()
y_pred_pca = rf_model.predict(X_test_pca)
end_time = time.time()
accuracy_pca = root_mean_squared_error(y_test_pca, y_pred_pca)
print("Среднеквадратичная ошибка после PCA с сохранением 90% дисперсии:", accuracy_pca)
print("Время предсказания с использованием PCA:", end_time - start_time)